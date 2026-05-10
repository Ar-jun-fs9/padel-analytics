# Padel Game Analytics — Shot Classification System

Computer Vision prototype for analyzing padel gameplay footage using YOLOv8, object tracking, and rule-based shot classification.

---
<!-- 
## Demo

> Add demo GIF / screenshots here

```md
![Demo](assets/demo.gif)
```

--- -->

## Executive Summary

This project implements a lightweight computer vision pipeline for analyzing padel match footage. The system detects and tracks players and ball movement, classifies shot types, and exports structured analytics.

### Core Features

- Player and ball detection using YOLOv8
- Multi-object tracking with stable player IDs
- Rule-based shot classification
- Shot analytics export (JSON / CSV)
- Annotated output video
- Simple analytics dashboard

### Supported Shot Types

- Forehand
- Backhand
- Smash
- Volley

### Important Note

The current prototype directly detects:
- Players
- Ball

Racket interaction is inferred using ball trajectory and player proximity because rackets are:
- small in CCTV footage,
- heavily occluded,
- not included in standard COCO detection classes.

---

## Quick Start

### Installation

```bash
git clone https://github.com/Ar-jun-fs9/padel-analytics.git
cd padel_analytics

python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

## Run the Project

### Synthetic Demo

```bash
python demo_run.py
```

### Real Video Processing

```bash
python main.py --video input.mp4
```

### With Live Preview

```bash
python main.py --video input.mp4 --show-preview
```

### Faster Processing

```bash
python main.py --video input.mp4 --skip-frames 2
```

### Launch Dashboard

```bash
python main.py --video input.mp4 --dashboard
```

---

## Outputs

The pipeline generates:

- `output/output_annotated.mp4`
- `output/shots.json`
- `output/shots.csv`
- `output/analytics_report.json`

---

## System Architecture

### Pipeline Overview

```text
Video Input
     │
     ▼
YOLOv8 Detection
(players + ball)
     │
     ▼
Multi-Object Tracking
(stable player IDs)
     │
     ▼
Ball Tracking
(position + velocity)
     │
     ▼
Shot Classification
(forehand/backhand/smash)
     │
     ▼
Analytics Engine
(counts + statistics)
     │
     ▼
JSON / CSV / Annotated Video
```

---

## Methodology

### 1. Object Detection

YOLOv8n is used for detecting:
- persons
- sports ball

The model is lightweight and works reasonably well without additional training.

---

### 2. Player Tracking

A lightweight SORT-style tracker maintains stable player IDs across frames using:
- bounding box overlap,
- center distance matching.

Court-zone filtering is also used to ignore spectators/background detections.

---

### 3. Ball Tracking

Ball positions are smoothed across frames to reduce jitter and estimate:
- velocity,
- direction,
- bounce behavior.

---

### 4. Shot Classification

Shot classification is rule-based.

The system uses:
- ball speed,
- movement direction,
- relative ball-player position,
- player court side.

Example:
- Ball on player's dominant side → likely forehand
- Ball crossing opposite side → likely backhand
- Fast downward motion → smash

---

## Why Rule-Based Classification?

A lightweight rule-based approach was chosen because:

1. No labeled padel shot dataset was available
2. The assignment timeline favored rapid prototyping
3. Rules are interpretable and easy to debug
4. No expensive model training was required
5. Easy threshold tuning during experimentation

### Future Improvement

With more labeled data:
- LSTM / Transformer models could be trained on trajectory sequences
- Pose estimation could improve shot recognition accuracy

---

## Why YOLOv8n?

YOLOv8n was selected because it is:
- lightweight,
- fast on CPU,
- easy to integrate,
- good enough for prototype-level detection.

Benefits:
- small model size,
- fast inference,
- no custom training required,
- easy upgrade path to larger models.

---

## Technology Stack

| Component | Technology |
|---|---|
| Detection | YOLOv8n |
| Tracking | Custom SORT-lite |
| Video Processing | OpenCV |
| Analytics | NumPy + Pandas |
| Dashboard | Flask |
| Language | Python 3.9+ |

---

## Performance

Tested on:
- synthetic gameplay footage,
- CCTV-style padel footage.

Approximate performance:
- ~8–12 FPS on CPU
- 720p video support
- Near real-time with frame skipping enabled

---

## Output Format

### JSON Output Example

```json
{
  "frame": 120,
  "timestamp": 4.0,
  "shot_type": "FOREHAND",
  "player_id": 1,
  "player_side": "left_court",
  "confidence": 0.87
}
```

### CSV Output

The same shot information is exported in CSV format for analysis in:
- Excel,
- Pandas,
- dashboards.

---

## Sample Annotated Output

The generated video contains:
- player bounding boxes,
- ball trail,
- live shot labels,
- analytics overlay.

> Add screenshot here

```md
![Sample Output](assets/sample_output.png)
```

---

## Repository Structure

```text
padel_analytics/
│
├── main.py
├── analyzer.py
├── demo_run.py
├── requirements.txt
│
├── utils/
│   ├── tracker.py
│   ├── ball_tracker.py
│   ├── shot_classifier.py
│   ├── analytics.py
│   └── visualizer.py
│
├── dashboard/
│   └── app.py
│
├── output/
│
└── assets/
```

---

## Validation

The system was manually tested for:
- player ID consistency,
- shot timing,
- JSON/CSV correctness,
- annotated video rendering.

Testing was performed on:
- synthetic gameplay sequences,
- real padel match footage.

---

## Failure Cases & Limitations

Current limitations include:

- No direct racket detection
- Ball misses during fast motion blur
- Occlusion issues behind players/walls
- Left-handed players may be misclassified
- Single-camera setup only
- No point/rally winner detection

---

## Detection Scope Clarification

### Detected Objects

| Object | Method |
|---|---|
| Player | YOLOv8 person detection |
| Ball | YOLOv8 sports ball detection |

### Not Directly Detected

| Object | Reason |
|---|---|
| Racket | Small + occluded + not in COCO |
| Court Lines | Not required for prototype |
| Net | Not required for current logic |

---

## Challenges Faced

Main challenges during development:

- Player ID switching
- Ball detection instability
- Motion blur
- Small object tracking
- Court-side assignment consistency

Solutions included:
- hybrid tracking logic,
- smoothing filters,
- court-zone heuristics.

---

## Future Improvements

Given more time, I would add:

- Custom-trained padel detector
- Racket detection
- Pose estimation
- Kalman filter ball prediction
- Court homography mapping
- Deep learning shot classifier
- Rally and scoring analytics

---

## Dependencies

Main libraries used:

- OpenCV
- Ultralytics YOLOv8
- NumPy
- Pandas
- Flask
- SciPy

Install using:

```bash
pip install -r requirements.txt
```

---

## Conclusion

This project demonstrates a practical end-to-end computer vision pipeline for sports analytics using lightweight and interpretable methods.

The focus was on:
- building a working prototype,
- modular system design,
- practical engineering tradeoffs,
- explainable shot classification.

While not production-ready, the system provides a strong foundation for future padel analytics research and development.

---

## Author

Arjun  
Layman AI Assignment — May 2026