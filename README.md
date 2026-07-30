# Face Detection, Tracking & Expression Recognition

> **University Final-Year Project** — Production-quality face analysis pipeline using MediaPipe BlazeFace, DeepSORT, and DeepFace with a Streamlit analytics dashboard.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Folder Structure](#folder-structure)
4. [Technology Stack](#technology-stack)
5. [System Requirements](#system-requirements)
6. [Installation](#installation)
7. [Running the Pipeline](#running-the-pipeline)
8. [Running the Dashboard](#running-the-dashboard)
9. [Configuration](#configuration)
10. [Function Reference](#function-reference)
11. [JSON Output Format](#json-output-format)
12. [Accuracy Improvement Tips](#accuracy-improvement-tips)
13. [Performance Optimisation Tips](#performance-optimisation-tips)
14. [Troubleshooting](#troubleshooting)
15. [Note on DeepFace Model Downloads](#note-on-deepface-model-downloads)

---

## Project Overview

This project builds a complete end-to-end video analysis system that:

- Detects every face in a video using **MediaPipe BlazeFace Short Range**
- Tracks each face across frames using **DeepSORT**
- Assigns **permanent, stable Track IDs** using face re-identification (ReID) via DeepFace ArcFace embeddings
- Recognises facial expressions (**Happy, Sad, Angry, Fear, Surprise, Neutral**) using **DeepFace**
- Saves the best face crop per person to disk
- Generates an annotated `output_video.mp4` with identical FPS, resolution, and frame count to the input
- Exports structured data to `result.json`
- Provides an interactive analytics dashboard via **Streamlit + Plotly**

---

## Features

| Feature | Details |
|---------|---------|
| Face Detection | MediaPipe BlazeFace Short Range (`model_selection=0`) |
| Multi-face tracking | DeepSORT — handles simultaneous multiple faces |
| Permanent IDs | Face ReID using ArcFace embeddings + cosine similarity |
| ID stability | IDs never change mid-track; person gets same ID on reappearance |
| Emotion recognition | DeepFace with temporal smoothing (rolling window, majority vote) |
| No Disgust | Disgust is automatically remapped to Neutral everywhere |
| Best face save | Only overwrites saved image if new crop is larger |
| Correct output | Same FPS, resolution, and frame count as input |
| Live display | `cv2.imshow()` with correct playback speed; press Q to quit |
| Dashboard | Streamlit + Plotly: KPIs, charts, gallery, timeline, search, heatmap |

---

## Folder Structure

```
FaceExpressionProject/
├── main.py                  ← Core detection/tracking/emotion pipeline
├── dashboard.py             ← Streamlit analytics dashboard (standalone)
├── requirements.txt         ← Python dependencies
├── README.md                ← This file
└── Outputs/                 ← Generated automatically by main.py
    ├── run001/
    │   ├── input_video.mp4      ← Copied input video
    │   ├── output_video.mp4     ← Annotated output video
    │   ├── result.json          ← Structured detection data
    │   └── captured_faces/      ← Best face crops saved by emotion
    ├── run002/
    └── ...
```

---

## Technology Stack

| Component | Library / Model | Version |
|-----------|----------------|---------|
| Face Detection | MediaPipe BlazeFace | ≥ 0.10.0 |
| Object Tracking | deep-sort-realtime | ≥ 1.3.2 |
| Face ReID | DeepFace + ArcFace | ≥ 0.0.93 |
| Emotion Recognition | DeepFace | ≥ 0.0.93 |
| DL Backend | TensorFlow | ≥ 2.13.0 |
| Video I/O | OpenCV | ≥ 4.8.0 |
| Dashboard UI | Streamlit | ≥ 1.28.0 |
| Dashboard Charts | Plotly | ≥ 5.17.0 |
| Numerics | NumPy, SciPy, Pillow | latest |

---

## System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| OS | Windows 10 64-bit | Windows 11 64-bit |
| CPU | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| RAM | 8 GB | 16 GB |
| GPU | — (CPU-only works) | NVIDIA GPU with CUDA 11+ |
| Disk | 5 GB free (model weights + output) | 10 GB free |
| Python | 3.9 | 3.10 / 3.11 |
| pip | ≥ 23.0 | latest |

---

## Installation

> All commands are for Windows. Run either **CMD** or **PowerShell** as Administrator.

### Step 1 — Clone / Download

Place all project files in a folder, e.g. `C:\FaceExpressionProject\`.

### Step 2 — Create Virtual Environment

**CMD:**
```cmd
cd C:\FaceExpressionProject
python -m venv venv
venv\Scripts\activate
```

**PowerShell:**
```powershell
cd C:\FaceExpressionProject
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell gives an execution-policy error:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Step 3 — Upgrade pip

**CMD / PowerShell:**
```cmd
python -m pip install --upgrade pip
```

### Step 4 — Install Dependencies

**CMD:**
```cmd
pip install -r requirements.txt
```

**PowerShell:**
```powershell
pip install -r requirements.txt
```

> **GPU users**: Install the CUDA-enabled TensorFlow separately:
> ```cmd
> pip install tensorflow[and-cuda]
> ```

### Step 5 — Prepare Input Video

Any common video format (mp4, avi, mkv) is supported. You do not need to move or rename it, just pass the path to the script in the next step.

---

## Running the Pipeline

**CMD:**
```cmd
venv\Scripts\activate
python main.py --video path\to\your\input_video.mp4
```

**PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
python main.py --video path\to\your\input_video.mp4
```

### What happens:
1. Creates a new `Outputs/runXXX/` folder and copies your video there.
2. Reads the video and prints its properties (FPS, resolution, frame count).
3. Opens a live preview window — **press Q** to quit early.
4. Processes every frame: detects, tracks, identifies, classifies emotions.
5. Saves face crops to `Outputs/runXXX/captured_faces/<Emotion>/person<ID>.jpg`.
6. Writes `Outputs/runXXX/output_video.mp4` (same FPS, same resolution, same frame count).
7. Writes `Outputs/runXXX/result.json` with all detection records.
8. Merges duplicate IDs automatically at the end and prints a summary.

---

## Running the Dashboard

The dashboard reads `result.json` — run `main.py` at least once first.

**CMD:**
```cmd
venv\Scripts\activate
streamlit run dashboard.py
```

**PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
streamlit run dashboard.py
```

Open your browser at `http://localhost:8501`. Use the sidebar to select the run you want to analyse.

### Dashboard Panels

| Panel | Description |
|-------|-------------|
| Overview KPIs | Total unique faces and per-emotion detection counts |
| Bar Chart | Detections per expression |
| Pie Chart | Expression share |
| Timeline | Detections over time (30-second bins) |
| Frame Scatter | Expression per frame number |
| Gallery | Face image grid, filterable by expression |
| Latest Detections | Last 20 records as a sortable table |
| Search by ID | Detailed view of one person: dominant emotion, frame range, image |
| Heatmap | Detection density by hour and day of week |

---

## Configuration

All tunable constants are at the top of `main.py`:

```python
# MediaPipe BlazeFace
BLAZE_CONFIDENCE   = 0.25     # lowered further to ensure absolutely all people are detected

# DeepSORT
DEEPSORT_MAX_AGE   = 40       # frames before a lost track is removed
DEEPSORT_N_INIT    = 1        # confirm track immediately so boundaries appear instantly

# Face ReID
REID_SIMILARITY_THRESHOLD = 0.75  # match threshold — same-person scores 0.80+ in same scene

# Emotion recognition
EMOTION_WINDOW     = 15       # larger window = more stable majority vote
EMOTION_EVERY_N    = 3        # run DeepFace every 3 frames per person
```

---

## Function Reference

### `main.py`

| Function | Module / Class | Description |
|----------|---------------|-------------|
| `ensure_dirs()` | module | Creates `captured_faces/<Emotion>/` sub-folders |
| `map_emotion(raw)` | module | Normalises DeepFace label → supported set; Disgust → Neutral |
| `cosine_similarity(a, b)` | module | Returns cosine similarity ∈ [0, 1] |
| `FaceDetector.__init__()` | `FaceDetector` | Initialises MediaPipe BlazeFace Short Range |
| `FaceDetector.detect(frame)` | `FaceDetector` | Runs BlazeFace; returns `[(x1,y1,x2,y2,conf)]` |
| `FaceDetector.close()` | `FaceDetector` | Releases MediaPipe resources |
| `FaceReID.get_embedding(crop)` | `FaceReID` | Extracts ArcFace 128-d embedding via DeepFace |
| `FaceReID.resolve(ds_id, crop)` | `FaceReID` | Maps DeepSORT ID → permanent ID via cosine similarity |
| `FaceReID.update_embedding(id, emb)` | `FaceReID` | Updates stored embedding with exponential moving average |
| `FaceReID.get_best_area(id)` | `FaceReID` | Returns area of best stored crop for an ID |
| `FaceReID.set_best_area(id, area, path)` | `FaceReID` | Updates best area and path for an ID |
| `EmotionSmoother.add(id, emotion)` | `EmotionSmoother` | Appends raw emotion to rolling window |
| `EmotionSmoother.get(id)` | `EmotionSmoother` | Returns majority-vote emotion from window |
| `EmotionSmoother.should_run(id)` | `EmotionSmoother` | Throttles DeepFace calls to every N frames |
| `analyze_emotion(crop)` | module | Runs DeepFace on 224×224 crop; returns normalised emotion |
| `save_best_face(...)` | module | Saves crop if its area exceeds stored best; returns path |
| `annotate_frame(...)` | module | Draws box, ID, emotion, timestamp on frame |
| `draw_overlay_info(...)` | module | Draws frame-count and FPS HUD in top-right corner |
| `main()` | module | Orchestrates the entire pipeline |

### `dashboard.py`

| Function | Description |
|----------|-------------|
| `load_data(path)` | Loads and normalises `result.json` into a DataFrame (cached 30 s) |
| `img_to_b64(path)` | Reads image file; returns base64 string for HTML embedding |
| `render_sidebar(df)` | Renders filter controls; returns `(sel_expr, sel_id, date_range)` |
| `render_kpis(df)` | Renders KPI card grid (total faces, per-emotion counts) |
| `render_charts(df)` | Renders Plotly bar chart and pie chart |
| `render_timeline(df)` | Renders area timeline and frame-level scatter chart |
| `render_gallery(df, expr_filter)` | Renders face image grid (one card per Track ID) |
| `render_latest(df)` | Renders last-20-detections dataframe |
| `render_search(df, sel_id)` | Renders per-ID detail view with image and distribution chart |
| `render_heatmap(df)` | Renders hour × day density heatmap |
| `main()` | Assembles all dashboard panels |

---

## JSON Output Format

```json
[
    {
        "track_id": 1,
        "timestamp": "2026-07-01 15:42:28",
        "frame_number": 325,
        "expression": "Happy",
        "face_image": "captured_faces/Happy/person1_325.jpg"
    },
    ...
]
```

One record is written per face per frame. Duplicate `(track_id, frame_number)` pairs are deduplicated before saving.

---

## Accuracy Improvement Tips

1. **Use a high-quality input video** — at least 720p, well-lit, faces close to camera.
2. **Increase `BLAZE_CONFIDENCE`** (e.g. `0.65`) to reduce false positives in crowded scenes.
3. **Increase `EMOTION_WINDOW`** (e.g. `10–15`) for more stable emotion labels in long videos.
4. **Decrease `EMOTION_EVERY_N`** (to `1`) for maximum accuracy at the cost of speed.
5. **Increase `REID_SIMILARITY_THRESHOLD`** (e.g. `0.75`) if different people are getting the same ID.
6. **Decrease `REID_SIMILARITY_THRESHOLD`** (e.g. `0.55`) if the same person keeps getting new IDs.
7. **Good lighting** is the single biggest factor in emotion accuracy — avoid harsh shadows.
8. **Face the camera** — profile faces are poorly classified by most emotion models.

---

## Performance Optimisation Tips

1. **GPU**: Install TensorFlow GPU (`pip install tensorflow[and-cuda]`) for 5–10× speedup on DeepFace.
2. **`EMOTION_EVERY_N`**: Increase to `5` or `10` to reduce DeepFace calls; emotion changes slowly.
3. **`DEEPSORT_N_INIT`**: Increase to `3` to reduce spurious short tracks.
4. **Frame skip**: For very long videos, process every 2nd frame (modify the frame loop in `main.py`).
5. **Lower resolution input**: If speed is critical, resize the frame before detection (while writing original to output).
6. **Batch processing**: For multiple videos, run `main.py` sequentially or in parallel processes.
7. **SSD/NVMe storage**: Writing `captured_faces/` images is faster on SSD.

---

## Troubleshooting

| Problem | Solution |
|---------|---------|
| `ModuleNotFoundError: No module named 'cv2'` | Run `pip install opencv-python` |
| `ModuleNotFoundError: No module named 'mediapipe'` | Run `pip install mediapipe` |
| `ModuleNotFoundError: No module named 'deep_sort_realtime'` | Run `pip install deep-sort-realtime` |
| `ModuleNotFoundError: No module named 'deepface'` | Run `pip install deepface` |
| `Cannot open video: 'input_video.mp4'` | Place `input_video.mp4` in the same folder as `main.py` |
| Output video plays too fast or too slow | Verify `cv2.CAP_PROP_FPS` is read correctly; check `fps` in terminal output |
| All faces labelled Neutral | DeepFace model not yet downloaded; wait for first-run download to complete |
| Same person gets multiple IDs | Lower `REID_SIMILARITY_THRESHOLD` (try `0.55`) |
| Different people share one ID | Raise `REID_SIMILARITY_THRESHOLD` (try `0.75`) |
| `streamlit: command not found` | Activate venv first or run `python -m streamlit run dashboard.py` |
| Dashboard shows "result.json not found" | Run `python main.py` first to generate the file |
| TensorFlow CUDA errors | Install CPU-only TF: `pip install tensorflow-cpu` |
| Low FPS in live window | Increase `EMOTION_EVERY_N` to reduce DeepFace calls |

---

## Note on DeepFace Model Downloads

DeepFace automatically downloads its model weight files on the **first run**.  
The following models will be downloaded:

| Model | Purpose | Approx Size |
|-------|---------|------------|
| ArcFace | Face ReID embeddings | ~250 MB |
| RetinaFace | Face detection backend for emotion | ~150 MB |
| Emotion model | Expression classification | ~50 MB |

**Total first-run download: ~450–600 MB**

These are saved to `~/.deepface/weights/` and reused on every subsequent run. Ensure you have a stable internet connection for the first run.

---

## Licence

This project was created as a university final-year demonstration project.  
Feel free to adapt and extend it for academic or research purposes.

---

## Author
**Segu Yoganand**
- GitHub: https://github.com/YoganandSegu
- Email: yoganandsegu0205@gmail.com

## Acknowledgements
- Thanks to my mentors and faculty.
- Thanks to the open-source community for the libraries and tools used.

---
⭐ If you found this project useful, please consider giving it a star!
