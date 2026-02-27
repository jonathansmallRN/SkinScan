# Data Dictionary
Project: SkinScan — Clinical Segmentation Framework  
Version: 1.0  
Setting: Research / Feasibility Prototype  

---

## 1. Purpose

This document defines the required and optional metadata fields for the SkinScan project.

All data validation scripts must enforce this schema before model training begins.

This file acts as the **data contract** between:

- Raw dataset
- Data preparation scripts
- Training pipeline
- Evaluation modules

No silent assumptions are permitted.

---

## 2. Required Fields

| Field Name   | Type   | Required | Description |
|--------------|--------|----------|-------------|
| image_id     | string | Yes      | Unique identifier for the image |
| image_path   | string | Yes      | Relative path to the .jpg image file |
| mask_path    | string | Yes      | Relative path to the .png mask file |
| split        | string | Yes      | Dataset split: train / val / test |

---

## 3. Strongly Recommended Fields (Leakage Prevention)

| Field Name   | Type   | Required | Description |
|--------------|--------|----------|-------------|
| patient_id   | string | Strongly Recommended | Unique identifier for patient. Required for gold-standard leakage prevention. |
| capture_date | string | Optional | ISO 8601 date of image capture (YYYY-MM-DD) |
| site_id      | string | Optional | Clinical site identifier |

## 3.1 Optional Quality Control Fields

These fields are optional but supported by the validation checklist.

| Field Name       | Type    | Required | Description |
|------------------|---------|----------|-------------|
| exclude          | boolean | Optional | Indicates image was rejected per ANNOTATION_SOP.md |
| exclude_reason   | string  | Optional | Free-text reason for rejection |
| irr_performed    | boolean | Optional | Indicates image was part of IRR subset |
| annotator_id     | string  | Optional | Identifier for annotator |
| sop_version      | string  | Optional | Version of SOP used during labeling |

If present, validation scripts must:
- Remove rows where `exclude=true`
- Log exclusion counts and reasons
- Surface IRR participation if available
---

## 4. Data Splitting Rules

### 4.1 Gold Standard
If `patient_id` exists:

- All splitting must occur at the patient level.
- A patient may appear in only one split.
- No patient_id may appear in multiple splits.

### 4.2 If patient_id Is Missing

Default behavior: **STRICT MODE**

- Training must stop with an error.
- Log message: "HIGH RISK DATA LEAKAGE — patient_id not available."

Optional behavior: **PERMISSIVE MODE (Research Only)**

- Allow image-level split.
- Log:
  - Explicit leakage warning
  - Split method used
  - Random seed
  - Image counts per split
- Generate `split_report.md`

No silent fallback is allowed.

---

## 5. Split Definitions

- train = Used for model training
- val = Used for validation during training
- test = Held-out final evaluation set

Splits must:

- Be reproducible
- Use a fixed random seed
- Be logged to disk

---

## 6. File Validation Requirements

Before training, scripts must verify:

1. Every `image_path` exists.
2. Every `mask_path` exists.
3. image_id matches mask base filename.
4. No duplicate image_id values.
5. No duplicate mask_path values.
6. No patient_id crosses splits (if present).

Any violation must halt execution.

---

## 7. Area Measurement Standard

Absolute area (cm²) is not supported.

Only pixel-based metrics are allowed:

Relative Area = Number of foreground pixels

Relative Change % =

Change % = ((Current Area − Baseline Area) / Baseline Area) × 100

No physical calibration claims may be made without fiducial marker protocol.

---

## 8. Known Dataset Limitations

- Fitzpatrick skin tone metadata may be unavailable.
- Lighting variability is uncontrolled.
- No standardized scale reference marker.
- No longitudinal guarantees in proxy datasets.

All limitations must be documented in MODEL_CARD.md.

---

## 9. Version Control

Any change to required fields must:

- Increment version number
- Update validation scripts
- Be logged in repository history

This ensures reproducibility and auditability.
