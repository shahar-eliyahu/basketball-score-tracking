# Basketball Score Tracking

A computer vision project for detecting basketball game elements and building toward automatic shot tracking and live score estimation from video.

## Project Goal

The long-term pipeline is:

```
Phone Camera / Recorded Video
        ↓
Object Detection
        ↓
Object Tracking
        ↓
Shot Detection
        ↓
Made / Missed
        ↓
Score Logic
        ↓
Live Scoreboard
```

The current stage focuses on building and training an object detector for:

- `basketball`
- `hoop`
- `player`

The phone is intended mainly as the video source, while inference can initially run on a laptop GPU.

## Dataset Sources

The development dataset was created by combining three basketball object-detection datasets from Roboflow, exported in YOLO11 format:

- `Basketball detection.v1i.yolov11`
- `Basketball Game Detections.v9i.yolov11`
- `Basketball.v1i.yolov11`

The original datasets used different class definitions, so their annotations were remapped into three unified classes:

- `basketball`
- `hoop`
- `player`

The original train / validation / test splits were not preserved. All samples were merged into a single pool, cleaned, and then divided into a new leakage-aware train / validation split.

### External Evaluation Datasets

Two additional Roboflow datasets are kept completely separate from model development and are reserved for final external evaluation.

**Dataset 04 — Basketball YOLO Dataset**

- Classes: `ball`, `hoop`, `player`
- License: CC BY 4.0
- Source: https://universe.roboflow.com/basketball-yolo-dataset/basketball-yolo-dataset/dataset/1

**Dataset 05 — Basketball Hoop, Ball and Player**

- Original classes: `3pt_area`, `ball`, `court`, `hoop`, `number`, `paint`, `player`
- License: CC BY 4.0
- Source: https://universe.roboflow.com/basketball-stat-tracker/basketball-hoop-ball-and-player/dataset/1

Only the original `test` split of each external dataset is reserved for final evaluation.

Dataset 04 already uses the required object classes.

Dataset 05 will be remapped for evaluation:

- `ball` → `basketball`
- `hoop` → `hoop`
- `player` → `player`

All other Dataset 05 classes will be ignored.

The external datasets are not used for training, validation, hyperparameter tuning, or model selection.

## Dataset Preparation

Three basketball object-detection datasets were merged into one unified pool. The original source `train` / `valid` / `test` splits were preserved only as metadata — a completely new split was created after cleaning.

### Unified Classes

All source labels were mapped into:

| ID | Class |
|----|-------|
| 0 | basketball |
| 1 | hoop |
| 2 | player |

Annotations such as `FG Attempt`, `FG Made`, `Ball in Basket`, and referee labels were excluded, since shot outcomes will later be determined using tracking and temporal logic rather than static labels.

### Data Cleaning Pipeline

1. **Dataset structure inspection** — verified image files, label files, and YOLO annotation format (`class_id x_center y_center width height`, normalized coordinates).
2. **Class mapping** — converted labels from all three source datasets to the unified three-class format.
3. **Dataset merge** — merged all source images into one temporary pool, preserving source dataset, original split, original filename, and merged filename in `source_manifest.csv`.
4. **Missing file validation** — checked images/labels for consistency.
   - Missing clean labels: `0`
   - Removal entries missing from dataset: `0`
5. **Class distribution analysis** — class distributions were verified after cleaning and after creating the final group-aware train / validation split.

   Classes were not forced to have equal annotation counts, since basketball scenes naturally contain more players than hoops or balls.

6. **Annotation visualization** — random samples visualized with ground-truth boxes to verify class correctness, box placement, annotation quality, and dataset relevance.
7. **Bounding-box QA**
   - Maximum annotations per image: `20`
   - Tiny-box review thresholds: basketball ≤ 6px, hoop ≤ 12px, player ≤ 8px
   - Same-class boxes with IoU ≥ 0.98 flagged as possible duplicates
   - Large boxes were analyzed but not used as a rejection criterion, since valid close-up images can naturally contain large objects
8. **Quality recheck** — suspicious samples were re-evaluated using multiple automatic quality checks.

   Samples were removed when they contained:

   - empty annotations
   - severe blur
   - duplicate annotations
   - exact duplicate images with conflicting annotations

   Tiny objects and images with many valid annotations were kept, since these can provide useful training information.

9. **Duplicate analysis**
   - **Exact duplicates** (SHA-256): exact duplicate images were detected and redundant copies with identical annotations were removed.
   - Exact duplicate groups with conflicting annotations were excluded during the final quality recheck.
   - **Near duplicates** (perceptual hashing): many visually similar images were found, as expected from repeated courts, camera angles, formations, and consecutive frames. These were kept because small visual changes can still provide useful training information.
10. **Blur analysis** — sharpness was measured using the variance of the Laplacian.
    - Blur threshold: `46.52`
    - Images below the final threshold: `199`
    - Strongly blurred samples were excluded during the final quality recheck.

### Final Clean Pool

| | Count |
|---|---:|
| Original images | 36,977 |
| Removed samples | 1,124 |
| **Clean samples** | **35,853** |

### Leakage-Aware Split

A normal random split could place multiple Roboflow variants of the same original image into different subsets.

To reduce this risk, samples were grouped using:

`source_dataset + original filename before ".rf."`

All variants belonging to the same original source image are assigned to the same split.

The final development dataset uses a reproducible group-aware `80 / 20` split:

| Split | Images |
|---|---:|
| Train | 28,682 |
| Validation | 7,171 |
| **Total** | **35,853** |

**Integrity checks:**

- Train / Validation overlap: `0`
- Unassigned samples: `0`
- Related Roboflow variants remain in the same split

No internal test set is used. Final evaluation is performed separately on Dataset 04 and Dataset 05.

## Final Dataset Structure

```
data/
├── training/
│   ├── dataset_01/
│   ├── dataset_02/
│   └── dataset_03/
│
├── merged/
│   └── basketball_dataset_temp/
│
├── processed/
│   └── basketball_dataset_final/
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── validation/
│       │   ├── images/
│       │   └── labels/
│       ├── data.yaml
│       └── final_manifest.csv
│
└── external_test/
    ├── dataset_04/
    └── dataset_05/
```

`final_manifest.csv` preserves the connection between every final image and its original source dataset and filename.

## Notebooks

- **`01_data_preparation.ipynb`** — dataset preparation, merging, class mapping, QA, duplicate analysis, blur analysis, cleaning, group-aware splitting, and final dataset creation.
- **`02_model_training.ipynb`** — YOLO26s training, validation, per-class analysis, error analysis, model selection, and final external evaluation.

## Model Direction

The first main detector planned for fine-tuning is **YOLO26s**.

During model development, evaluation will use the internal validation set.

Metrics include:

- Precision
- Recall
- mAP50
- mAP50-95

Special attention will be given to **basketball recall**, since missed ball detections can strongly affect later tracking and shot-detection stages.

Once model development is complete, the selected model will be evaluated separately on:

- External Test 1 — Dataset 04
- External Test 2 — Dataset 05

These datasets remain unseen during training and model selection.

## Next Steps

```
Load final development dataset
        ↓
Visual sanity check
        ↓
Train YOLO26s
        ↓
Evaluate on Validation
        ↓
Per-class analysis
        ↓
Error analysis
        ↓
Improve detector
        ↓
Select final model
        ↓
External Test 1
        ↓
External Test 2
        ↓
Object tracking
        ↓
Shot detection
        ↓
Made / missed logic
        ↓
Live scoreboard
```

## Tech Stack

- Python
- Jupyter Notebook
- OpenCV
- NumPy
- PyTorch
- Ultralytics YOLO
- CUDA / NVIDIA GPU