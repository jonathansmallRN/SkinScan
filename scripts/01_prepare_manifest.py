"""
scripts/01_prepare_manifest.py
SkinScan — Clinical Segmentation Framework
Version: 1.0

# Governance: Enforces patient-level splitting and validation firewall per DATA_DICTIONARY.

PURPOSE
-------
Build data/manifest.jsonl — the contract between raw data and the training
pipeline. Acts as a clinical firewall: nothing reaches model training until
every file pair, schema field, and split assignment has been verified.

GROUND TRUTH SOURCES
--------------------
- DATA_DICTIONARY.md  §2 Required Fields, §4 Splitting Rules,
                      §5 Split Definitions, §6 File Validation
- ANNOTATION_SOP.md   §5 Rejection Criteria, §6 Output Format

LIMITATIONS (required disclosure — this is a research prototype)
----------------------------------------------------------------
- Absolute area (cm²) is NOT supported. Pixel-only metrics only.
- No clinical diagnostic claims are made.
- If patient_id is absent and --permissive is used, metrics may be inflated.
- IRR may not have been performed; document in MODEL_CARD.md if so.

USAGE
-----
# Strict mode (default — fails if patient_id is missing):
python scripts/01_prepare_manifest.py

# Permissive mode (research only — image-level split with full warnings):
python scripts/01_prepare_manifest.py --permissive

# Custom paths:
python scripts/01_prepare_manifest.py \
    --images_dir data/raw/images \
    --masks_dir  data/raw/masks \
    --metadata   data/metadata.csv \
    --seed 42 \
    --val_ratio 0.15 \
    --test_ratio 0.15
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

# Import the validation firewall from src/data_utils.py.
# This must pass before any manifest is written to disk.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_utils import run_validation_checklist  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("data")
MANIFEST_PATH = OUTPUT_DIR / "manifest.jsonl"
LOG_PATH = OUTPUT_DIR / "manifest_build.log"


# ---------------------------------------------------------------------------
# Logging — writes to both console and data/manifest_build.log
# ---------------------------------------------------------------------------
def _setup_logging() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_PATH, mode="w"),
        ],
    )


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# STEP 1 — Scan raw folders and build base file-pair DataFrame
# Source: DATA_DICT §6 rules 1–3, SOP §6
#
# Clinical why: a mask paired to the wrong image means the model learns
# to segment the wrong region. We catch this before anything else runs.
# ---------------------------------------------------------------------------
def build_file_pairs(images_dir: Path, masks_dir: Path) -> pd.DataFrame:
    """
    Walk images_dir for .jpg files and match each to a .png mask in masks_dir
    using identical base filenames (e.g., ISIC_0001234.jpg → ISIC_0001234.png).

    Raises RuntimeError if any image lacks a paired mask or vice versa.
    """
    # Scan for both .jpg and .jpeg — ISIC and other clinical datasets
    # commonly use both extensions for identical file types.
    images = {
        p.stem: p
        for ext in ("*.jpg", "*.jpeg")
        for p in sorted(images_dir.glob(ext))
    }
    masks = {p.stem: p for p in sorted(masks_dir.glob("*.png"))}

    if not images:
        raise RuntimeError(f"No .jpg or .jpeg images found in {images_dir}")
    if not masks:
        raise RuntimeError(f"No .png masks found in {masks_dir}")

    # Unpaired images → hard error (DATA_DICT §6 rule 1 & 2)
    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    errors = []
    if missing_masks:
        errors.append(f"Images with NO matching mask ({len(missing_masks)}): {missing_masks[:10]}")
    if missing_images:
        errors.append(f"Masks with NO matching image ({len(missing_images)}): {missing_images[:10]}")
    if errors:
        raise RuntimeError("FILE PAIRING FAILURE:\n  " + "\n  ".join(errors))

    paired_stems = sorted(set(images) & set(masks))
    log.info(f"Found {len(paired_stems)} matched image/mask pairs.")

    rows = []
    for stem in paired_stems:
        rows.append({
            "image_id": stem,
            # Store ONLY the filename (not the full path).
            # The validator joins image_root / image_path at check time,
            # so storing the full path here would create a doubled path like:
            # data/raw/images/ + data/raw/images/ISIC_x.jpg → "missing image".
            # Source: DATA_DICT §6 rules 1–2 (validator path-join contract).
            "image_path": images[stem].name,
            "mask_path": masks[stem].name,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# STEP 2 — Merge optional metadata
# Source: DATA_DICT §2 Required Fields, §3 Strongly Recommended Fields
#
# Clinical why: patient_id is the only field that enables safe splitting.
# Without it we cannot guarantee a patient doesn't appear in both train
# and test, which would silently inflate every performance metric.
# ---------------------------------------------------------------------------
def merge_metadata(df: pd.DataFrame, metadata_path: Path) -> pd.DataFrame:
    """
    If metadata.csv exists, merge it onto the file-pair DataFrame on image_id.
    Carries patient_id, capture_date, site_id, exclude, exclude_reason, and
    any other columns present (future-proof).

    If metadata file is absent, log a warning and continue with base fields only.
    """
    if not metadata_path.exists():
        log.warning(
            f"Metadata file not found: {metadata_path}. "
            "Proceeding with file pairs only. "
            "patient_id will be unavailable — leakage check (STEP 3) will apply."
        )
        return df

    meta = pd.read_csv(metadata_path)
    log.info(f"Metadata loaded: {len(meta)} rows, columns: {list(meta.columns)}")

    if "image_id" not in meta.columns:
        raise RuntimeError(
            "metadata.csv must contain an 'image_id' column to merge on. "
            "Rename the identifier column to 'image_id'."
        )

    merged = df.merge(meta, on="image_id", how="left")
    log.info(f"Post-merge shape: {merged.shape}. Columns: {list(merged.columns)}")
    return merged


# ---------------------------------------------------------------------------
# STEP 3 — Assign splits with leakage prevention
# Source: DATA_DICT §4.1 (patient-level), §4.2 (STRICT/PERMISSIVE)
#
# Clinical why: if the same patient's wound photo appears in both train
# and test, the model recognises the patient's skin, not the lesion.
# Performance metrics become meaningless for unseen patients.
# ---------------------------------------------------------------------------
def assign_splits(
    df: pd.DataFrame,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    permissive: bool,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Assign each row a split value: train / val / test.

    Routes to patient-level or image-level splitting based on patient_id
    availability and the permissive flag.
    """
    # Defensive ratio validation — catch misconfiguration before any data is split.
    if not (0 < val_ratio < 1):
        raise ValueError(f"val_ratio must be between 0 and 1 (got {val_ratio})")
    if not (0 < test_ratio < 1):
        raise ValueError(f"test_ratio must be between 0 and 1 (got {test_ratio})")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError(
            f"val_ratio + test_ratio must be < 1.0 "
            f"(got {val_ratio} + {test_ratio} = {val_ratio + test_ratio:.2f}). "
            f"No data would remain for training."
        )
    patient_id_ok = (
        "patient_id" in df.columns
        and df["patient_id"].notna().sum() > 0
    )

    if patient_id_ok:
        log.info("patient_id found — using patient-level split (DATA_DICT §4.1).")
        df = _patient_level_split(df, seed, val_ratio, test_ratio)
        grouping_key = "patient_id (gold standard)"
    else:
        # No patient_id — enforce STRICT or PERMISSIVE per DATA_DICT §4.2
        if not permissive:
            raise RuntimeError(
                "\n"
                "╔══════════════════════════════════════════════════════════════╗\n"
                "║         HIGH RISK DATA LEAKAGE — STRICT MODE HALT           ║\n"
                "╠══════════════════════════════════════════════════════════════╣\n"
                "║  patient_id is not available in the manifest.               ║\n"
                "║  Patient-level split cannot be performed.                   ║\n"
                "║  Without it, the same patient may appear in both train      ║\n"
                "║  and test sets — inflating all performance metrics.         ║\n"
                "║                                                             ║\n"
                "║  SOURCE: DATA_DICTIONARY.md §4.2 (STRICT MODE default)     ║\n"
                "║                                                             ║\n"
                "║  RESOLUTION:                                                ║\n"
                "║    1. Add patient_id to data/metadata.csv, OR              ║\n"
                "║    2. Re-run with --permissive (research use only).         ║\n"
                "╚══════════════════════════════════════════════════════════════╝\n"
            )

        log.warning(
            "\n⚠  HIGH RISK DATA LEAKAGE — PERMISSIVE MODE ACTIVE\n"
            "   patient_id absent. Image-level split applied.\n"
            "   Results from this run are NOT reliable for clinical assessment.\n"
            "   Source: DATA_DICTIONARY.md §4.2\n"
        )
        df = _image_level_split(df, seed, val_ratio, test_ratio)
        grouping_key = "image-level (NO patient_id — HIGH RISK)"

    _write_split_report(df, seed, grouping_key, permissive, output_dir)
    return df


def _patient_level_split(
    df: pd.DataFrame, seed: int, val_ratio: float, test_ratio: float
) -> pd.DataFrame:
    """
    Split at the patient level using GroupShuffleSplit so no patient
    crosses the train/val/test boundary. DATA_DICT §4.1.
    """
    patients = df["patient_id"].values
    indices = df.index.values

    # First carve out test set from all data
    test_size = test_ratio
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss_test.split(indices, groups=patients))

    # Then carve val from remaining train+val
    val_relative = val_ratio / (1.0 - test_ratio)
    trainval_patients = patients[trainval_idx]
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_relative, random_state=seed)
    train_sub_idx, val_sub_idx = next(
        gss_val.split(trainval_idx, groups=trainval_patients)
    )

    train_idx = trainval_idx[train_sub_idx]
    val_idx = trainval_idx[val_sub_idx]

    df = df.copy()
    df.loc[df.index[train_idx], "split"] = "train"
    df.loc[df.index[val_idx], "split"] = "val"
    df.loc[df.index[test_idx], "split"] = "test"
    return df


def _image_level_split(
    df: pd.DataFrame, seed: int, val_ratio: float, test_ratio: float
) -> pd.DataFrame:
    """
    Image-level random split — RESEARCH ONLY.
    Only reached in PERMISSIVE mode. All warnings already logged.
    DATA_DICT §4.2 optional behavior.
    """
    df = df.copy()
    train_val, test = train_test_split(df, test_size=test_ratio, random_state=seed)
    val_relative = val_ratio / (1.0 - test_ratio)
    train, val = train_test_split(train_val, test_size=val_relative, random_state=seed)

    df.loc[train.index, "split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"
    return df


# ---------------------------------------------------------------------------
# STEP 4 — Write split_report.md
# Required by DATA_DICT §4.2 and project non-negotiable standards.
# ---------------------------------------------------------------------------
def _write_split_report(
    df: pd.DataFrame,
    seed: int,
    grouping_key: str,
    permissive: bool,
    output_dir: Path,
) -> None:
    split_counts = df["split"].value_counts().to_dict()
    mode_label = "PERMISSIVE (RESEARCH ONLY)" if permissive else "STRICT"
    warning_block = (
        "\n> ⚠ **HIGH RISK DATA LEAKAGE WARNING**\n"
        "> `patient_id` was not available. Image-level split was used.\n"
        "> Model metrics from this split are NOT reliable for clinical use.\n"
        "> Source: DATA_DICTIONARY.md §4.2\n"
        if permissive else ""
    )
    counts_table = "\n".join(
        f"| {s} | {split_counts.get(s, 0)} |"
        for s in ["train", "val", "test"]
    )

    report = f"""# SkinScan Split Report
{warning_block}
| Field | Value |
|---|---|
| Timestamp | {datetime.utcnow().isoformat()}Z |
| Mode | {mode_label} |
| Random seed | {seed} |
| Grouping key | {grouping_key} |
| Total images | {len(df)} |

## Split Counts

| Split | Images |
|---|---|
{counts_table}

## Limitations

- Absolute area (cm²) is not supported. Pixel-based metrics only. DATA_DICT §7.
- No clinical diagnostic claims are made by this split.
- If patient_id was absent, document this limitation in MODEL_CARD.md.

*Source rules: DATA_DICTIONARY.md §4, §5*
"""
    (output_dir / "split_report.md").write_text(report)
    log.info(f"split_report.md written to {output_dir / 'split_report.md'}")


# ---------------------------------------------------------------------------
# STEP 5 — Write manifest.jsonl
# Source: DATA_DICT §2 Required Fields
# ---------------------------------------------------------------------------
def write_manifest(df: pd.DataFrame, manifest_path: Path) -> None:
    """
    Write the validated DataFrame to a line-delimited JSON file.
    Each line is one image record. Required fields guaranteed present
    by this point (validated by run_validation_checklist).
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(record) + "\n")
    log.info(f"manifest.jsonl written: {len(df)} records → {manifest_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SkinScan manifest builder — clinical data firewall."
    )
    p.add_argument("--images_dir", default="data/raw/images",
                   help="Directory containing .jpg images.")
    p.add_argument("--masks_dir", default="data/raw/masks",
                   help="Directory containing .png masks.")
    p.add_argument("--metadata", default="data/metadata.csv",
                   help="Optional CSV with patient_id and QC fields.")
    p.add_argument("--output_manifest", default=str(MANIFEST_PATH),
                   help="Output path for manifest.jsonl.")
    p.add_argument("--seed", type=int, default=42,
                   help="Fixed random seed for reproducible splits.")
    p.add_argument("--val_ratio", type=float, default=0.15,
                   help="Fraction of data for validation split.")
    p.add_argument("--test_ratio", type=float, default=0.15,
                   help="Fraction of data for test split.")
    p.add_argument("--permissive", action="store_true", default=False,
                   help=(
                       "Allow image-level split if patient_id is missing. "
                       "RESEARCH USE ONLY. Writes HIGH RISK warning to split_report.md."
                   ))
    return p.parse_args()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    _setup_logging()
    args = parse_args()

    log.info("=" * 70)
    log.info("SKINSCAN MANIFEST BUILDER — START")
    log.info(f"Timestamp  : {datetime.utcnow().isoformat()}Z")
    log.info(f"Images dir : {args.images_dir}")
    log.info(f"Masks dir  : {args.masks_dir}")
    log.info(f"Metadata   : {args.metadata}")
    log.info(f"Seed       : {args.seed}")
    log.info(f"Val ratio  : {args.val_ratio}")
    log.info(f"Test ratio : {args.test_ratio}")
    log.info(f"Mode       : {'PERMISSIVE' if args.permissive else 'STRICT'}")
    log.info("=" * 70)

    # STEP 1 — Pair images with masks
    df = build_file_pairs(
        images_dir=Path(args.images_dir),
        masks_dir=Path(args.masks_dir),
    )

    # STEP 2 — Merge metadata (patient_id, QC flags, etc.)
    df = merge_metadata(df, metadata_path=Path(args.metadata))

    # STEP 3 — Assign splits with leakage prevention
    df = assign_splits(
        df=df,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        permissive=args.permissive,
        output_dir=OUTPUT_DIR,
    )

    # Resolve to absolute paths now, before passing anywhere.
    # If someone runs this script from a different working directory,
    # relative paths like "data/raw/images" would resolve to the wrong
    # location. Absolute paths are unambiguous regardless of CWD.
    images_dir_abs = str(Path(args.images_dir).resolve())
    masks_dir_abs = str(Path(args.masks_dir).resolve())

    # STEP 4 — Stage manifest to disk before validation.
    # The validator (run_validation_checklist) expects a file path, not an
    # in-memory DataFrame. We write a staging file first, validate against it,
    # then write the final manifest from the cleaned DataFrame the checklist
    # returns. This keeps unvalidated data off the canonical output path.
    staging_manifest = OUTPUT_DIR / "_manifest_precheck.jsonl"
    write_manifest(df, manifest_path=staging_manifest)

    log.info("Running validation checklist — clinical firewall...")
    df_validated = run_validation_checklist(
        manifest_path=str(staging_manifest),
        image_root=images_dir_abs,
        mask_root=masks_dir_abs,
        expected_seed=args.seed,
        permissive=args.permissive,
        output_dir=str(OUTPUT_DIR),
    )

    # Remove staging file — only the validated manifest belongs on disk.
    try:
        staging_manifest.unlink()
        log.info(f"Staging manifest removed: {staging_manifest}")
    except OSError as e:
        log.warning(f"Could not remove staging manifest: {e}")

    # STEP 5 — All checks passed. Write final manifest from validated DataFrame.
    # df_validated may have fewer rows than df if excluded images were filtered
    # by run_validation_checklist (SOP §5 reject criteria).
    write_manifest(df_validated, manifest_path=Path(args.output_manifest))

    log.info("=" * 70)
    log.info("MANIFEST BUILD COMPLETE — training pipeline may proceed.")
    log.info(f"  Records   : {len(df_validated)} (after exclusion filter)")
    log.info(f"  Manifest  : {args.output_manifest}")
    log.info(f"  Report    : {OUTPUT_DIR / 'split_report.md'}")
    log.info(f"  Log       : {LOG_PATH}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
