# Annotation SOP — Clinical Image Segmentation
**Project:** SkinScan (Research/Feasibility Prototype)  
**Version:** 1.0  
**Owner:** Jonathan Small, RN (Nursing Informatics)

---

## 1. Objective
Standardize segmentation labels to produce consistent, high-quality training data for a MONAI-based pipeline. This SOP prioritizes **repeatable clinical boundary rules** over aesthetic smoothing.

---

## 2. Scope (v1)
- **Task:** Binary segmentation (foreground vs background)
- **Data type:** 2D images (.jpg) with masks (.png)
- **Proxy dataset:** ISIC dermoscopic lesion images (for pipeline validation)

**Not in scope (v1):**
- Diagnostic labeling or staging claims
- Absolute size measurement (cm²)
- Multi-class tissue labeling

---

## 3. Label Definitions (Binary Segmentation)
- **Foreground (Class 1):** The *visible* lesion/target region intended for segmentation.
- **Background (Class 0):** Everything else (normal skin, surrounding tissue, artifacts, markers).

**Rule:** Label only what is visible. Do not infer boundaries beneath occlusion, glare, or shadow.

---

## 4. Boundary Rules
1. **Visible Edge Priority**  
   Trace the transition from abnormal pigment/texture to more uniform surrounding skin.

2. **Conservative Estimation**  
   If the border is blurred (e.g., inflammation/erythema, low contrast), default to the most clearly visible edge.

3. **Inclusion**  
   Include all contiguous regions that are clearly part of the target lesion region (continuous islands connected by visible lesion tissue).

4. **Exclusion (Always Exclude)**
   - Medical tape, gauze, bandage edges
   - Rulers/fiducial markers
   - Shadows and glare artifacts that fall outside the clearly visible lesion
   - Hair strands (do not trace hairs; label lesion boundary underneath if visible)

5. **Multiple Targets in One Image**
   - v1 default: label the **single primary lesion** (largest / most central).
   - If multiple lesions must be labeled, document as an exception in the annotation log.

---

## 5. Image Quality & Rejection Criteria
Reject an image from training if any of the following apply:
1. **Occlusion:** >30% of the target region is obscured (fingers, tape, tools).
2. **Blur:** Focus is too poor to identify the lesion-to-skin transition.
3. **Privacy risk:** PII present (faces, name bands, unique tattoos, identifying marks).
4. **Corruption:** File is unreadable or the mask cannot be aligned.

**Required logging:** If rejected, record `exclude=true` and an `exclude_reason` in the annotation log.

---

## 6. Output Format Requirements
- Masks must be **single-channel PNG**
- Foreground pixel value = **1**
- Background pixel value = **0**
- Mask must match image base filename

Example:  
`ISIC_0001234.jpg` → `ISIC_0001234.png`

---

## 7. Quality Control (QC)
### 7.1 Self-QC (every image)
Confirm:
- Mask follows visible boundary
- No inclusion of tape/ruler/major glare
- No holes unless clearly background

### 7.2 Inter-Rater Reliability (IRR) — if two annotators available
- Double-label **10%** of images
- Compute Dice Similarity Coefficient (DSC) between annotators

**Target DSC:** ≥ 0.85  
If DSC < 0.85:
- adjudicate disagreements on that subset
- update SOP with clarified edge-case rules
- document changes in version history

**If only one annotator exists:** Document IRR as “not performed” and treat as a limitation in the Model Card.

---

## 8. Version History
- v1.0: Initial binary segmentation SOP (ISIC proxy dataset)
