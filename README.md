# Basketball Score Tracking

A computer vision project for detecting and tracking basketball game elements, with the long-term goal of automatic shot detection, made / missed classification, score logic, and a live scoreboard from video.

## Project Goal

The planned system pipeline is:

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

The project currently supports detection and initial object tracking for:

- `basketball`
- `hoop`
- `player`

The phone is intended mainly as the future video source, while inference can initially run on a laptop GPU.

## Current Status

The project currently includes:

- A cleaned and unified basketball detection dataset
- Leakage-aware train / validation splitting
- Incremental YOLO26n training with checkpoint resume
- Validation and per-class performance analysis
- Confidence-threshold analysis
- Detection error analysis
- Basketball size-based recall analysis
- Independent tracking for basketballs, hoops, and players
- ByteTrack-based tracking as the current baseline
- Automatic loading of class-specific confidence thresholds
- Tracking statistics for comparing future tracker configurations

The detector continues to be trained incrementally while the tracking and system pipeline are developed in parallel.

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

Dataset 04 already uses the required object classes. For Dataset 05, only the relevant classes are mapped:

- `ball` → `basketball`
- `hoop` → `hoop`
- `player` → `player`

All other Dataset 05 classes are ignored.

The external evaluation data is not used for training, validation, hyperparameter tuning, or model selection.

## Dataset Preparation

Three basketball object-detection datasets were merged into one unified pool. The original source `train` / `valid` / `test` splits were preserved only as metadata — a completely new split was created after cleaning.

### Unified Classes

| ID | Class |
|----|-------|
| 0 | basketball |
| 1 | hoop |
| 2 | player |

Annotations such as `FG Attempt`, `FG Made`, `Ball in Basket`, and referee labels were excluded, since shot outcomes will later be determined using tracking and temporal logic rather than static image labels.

### Data Cleaning Pipeline

1. **Dataset structure inspection** — verified image files, label files, and YOLO annotation format (`class_id x_center y_center width height`, normalized coordinates).
2. **Class mapping** — converted labels from all three source datasets to the unified three-class format.
3. **Dataset merge** — merged all source images into one temporary pool, preserving source dataset, original split, original filename, and merged filename in `source_manifest.csv`.
4. **Missing file validation**
   - Missing clean labels: `0`
   - Removal entries missing from dataset: `0`
5. **Class distribution analysis** — class distributions were verified after cleaning and after the final split. Classes were not forced to have equal annotation counts, since basketball scenes naturally contain more players than hoops or balls.
6. **Annotation visualization** — random samples were visualized with ground-truth boxes to verify class correctness, box placement, annotation quality, and dataset relevance.
7. **Bounding-box QA**
   - Maximum annotations per image: `20`
   - Tiny-box review thresholds: basketball ≤ 6px, hoop ≤ 12px, player ≤ 8px
   - Same-class boxes with IoU ≥ 0.98 flagged as possible duplicates
   - Large boxes were analyzed but not used as a rejection criterion, since valid close-up images can naturally contain large objects
8. **Quality recheck** — samples were removed when they contained:
   - empty annotations
   - severe blur
   - duplicate annotations
   - exact duplicate images with conflicting annotations

   Tiny objects and images with many valid annotations were kept, since they provide useful training information.
9. **Duplicate analysis**
   - **Exact duplicates** (SHA-256): redundant copies with identical annotations were removed; groups with conflicting annotations were excluded during the quality recheck.
   - **Near duplicates** (perceptual hashing): visually similar images were kept, since repeated courts, camera angles, and consecutive frames still provide useful variation.
10. **Blur analysis** — sharpness was measured using the variance of the Laplacian.
    - Blur threshold: `46.52`
    - Images below the final threshold: `199`

### Final Clean Pool

| | Count |
|---|---:|
| Original images | 36,977 |
| Removed samples | 1,124 |
| **Clean samples** | **35,853** |

### Leakage-Aware Split

A normal random split could place multiple Roboflow variants of the same original image into different subsets. To reduce this risk, samples were grouped using:

`source_dataset + original filename before ".rf."`

All variants belonging to the same original source image are assigned to the same split. The final development dataset uses a reproducible group-aware `80 / 20` split:

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

## Model Training

The current detector is **YOLO26n**.

Training is performed incrementally in short sessions rather than one continuous run. The training workflow supports:

- Resume from `last.pt`
- Best-model preservation using `best.pt`
- Periodic checkpoints
- Global early stopping across training sessions
- Continued training toward a maximum epoch target

This allows training to continue overnight while development of the tracking and shot-detection pipeline continues during the day.

## Model Evaluation

Model development is evaluated only on the internal validation set. The evaluation workflow includes:

- Precision, Recall, mAP50, mAP50-95
- Per-class evaluation
- Training-progress analysis
- Confidence-threshold analysis
- Confusion matrix
- Detection error examples
- Basketball recall by object size

Special attention is given to **basketball recall**, since missed ball detections can directly interrupt tracking and later shot-detection logic.

Once model development is complete, the selected model will be evaluated separately on:

- External Test 1 — Dataset 04
- External Test 2 — Dataset 05

### Class-Specific Confidence Thresholds

The evaluation notebook calculates a confidence threshold for each class based on validation performance and saves them automatically to:

`config/selected_thresholds.json`

The tracking system loads this file on startup, so thresholds can be recalculated whenever a newer `best.pt` is evaluated without hardcoding them inside the tracking code.

## Object Tracking

Tracking is implemented separately for each object type:

```
YOLO Detector
      ↓
Detections
      ↓
ObjectTracker
      ↓
┌────────────────┬────────────────┬────────────────┐
↓                ↓                ↓
BallTracker      HoopTracker      PlayerTracker
```

The detector runs only once per video frame. The resulting detections are split by class and passed to independent trackers.

| Tracker | Current algorithm |
|---|---|
| BallTracker | ByteTrack (baseline) |
| HoopTracker | ByteTrack (baseline) |
| PlayerTracker | ByteTrack (baseline) |

Each tracker is implemented independently so different algorithms or configurations can later be tested per object type. This matters because the tracking requirements differ:

- **Basketballs** are small, fast, and can disappear between frames.
- **Players** are larger but can overlap and occlude each other.
- **Hoops** are relatively stable but can shift within the image when the camera moves.

### Tracking Analysis

The system records tracking statistics such as:

- Frames processed
- Frames containing detections
- Total detections
- Unique Track IDs
- Frames with detections but no returned track
- Longest continuous track

These measurements provide a baseline for comparing tracking algorithms and configurations.

## Project Structure

```
basketball_score_tracking/
│
├── 01_data_preparation.ipynb
├── 02_model_training.ipynb
├── 03_model_evaluation.ipynb
├── 04_final_test_evaluation.ipynb
│
├── main.py
│
├── config/
│   └── selected_thresholds.json
│
├── src/
│   ├── __init__.py
│   ├── detector.py
│   ├── video_processor.py
│   │
│   └── tracking/
│       ├── __init__.py
│       ├── object_tracker.py
│       ├── ball_tracker.py
│       ├── hoop_tracker.py
│       └── player_tracker.py
│
├── data/
├── results/
├── runs/
├── videos/
├── .gitignore
└── README.md
```

Large datasets, videos, trained model weights, and generated training outputs are excluded from Git.

### Data Directory

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

- **`01_data_preparation.ipynb`** — dataset inspection, merging, class remapping, cleaning, duplicate and blur analysis, group-aware train / validation split.
- **`02_model_training.ipynb`** — incremental YOLO26n training, checkpoint resume and management, global early stopping.
- **`03_model_evaluation.ipynb`** — validation metrics, per-class evaluation, training progress, confidence-threshold analysis, confusion matrix, error analysis, basketball size-based recall, and export of class-specific thresholds.
- **`04_final_test_evaluation.ipynb`** — reserved for final evaluation on the external test datasets; not used during normal model development.

## Current Development Pipeline

```
Improve YOLO26n detector
        ↓
Evaluate current best.pt
        ↓
Update confidence thresholds
        ↓
Run object tracking
        ↓
Compare tracker performance
        ↓
Improve basketball tracking
        ↓
Ball trajectory analysis
        ↓
Shot detection
        ↓
Made / missed classification
        ↓
Score logic
        ↓
Live camera input
        ↓
Live scoreboard
```

Detector training and system development continue in parallel.

## Next Steps

1. Evaluate and compare tracking approaches for each object type.
2. Improve basketball track continuity when detections are temporarily lost.
3. Store short basketball trajectories by Track ID.
4. Detect shot attempts using ball motion relative to the hoop.
5. Determine made / missed outcomes.
6. Implement score logic.
7. Support live camera input.
8. Build a live scoreboard interface.

## Tech Stack

- Python
- Jupyter Notebook
- PyCharm
- OpenCV
- NumPy
- Pandas
- PyTorch
- Ultralytics YOLO
- ByteTrack
- CUDA / NVIDIA GPU