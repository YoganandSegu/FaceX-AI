"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   Face Detection · Tracking · Expression Recognition Pipeline               ║
║   ─────────────────────────────────────────────────────────────────────     ║
║   Detection  : MediaPipe BlazeFace Short Range (model_selection=0)          ║
║   Tracking   : DeepSORT (deep-sort-realtime)                                ║
║   ReID       : DeepFace ArcFace embeddings + cosine similarity              ║
║   Emotion    : DeepFace + temporal rolling-window smoothing                 ║
║   Output     : output_video.mp4  ·  result.json  ·  captured_faces/         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import json
import time
import logging
import warnings
from collections import deque, Counter
from datetime import datetime

# ── Suppress noisy TF / DeepFace logs before imports ─────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("deepface").setLevel(logging.ERROR)

# ── Third-party ───────────────────────────────────────────────────────────────
import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
from deep_sort_realtime.deepsort_tracker import DeepSort

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these values to tune behaviour
# ─────────────────────────────────────────────────────────────────────────────
INPUT_VIDEO        = "input_video.mp4"
OUTPUT_VIDEO       = "output_video.mp4"
RESULT_JSON        = "result.json"
FACES_DIR          = "captured_faces"

# MediaPipe BlazeFace
BLAZE_CONFIDENCE   = 0.25          # lowered further to ensure absolutely all people are detected

# DeepSORT
DEEPSORT_MAX_AGE   = 40            # frames before a lost track is removed
DEEPSORT_N_INIT    = 1             # confirm track immediately so boundaries appear instantly
DEEPSORT_MAX_IOU   = 0.7

# Face ReID
REID_SIMILARITY_THRESHOLD  = 0.75   # match threshold — same-person scores 0.80+ in same scene
REID_GALLERY_ENRICH_MIN    = 0.85   # only add to gallery when very confident (prevents snowball)
REID_AMBIGUITY_MARGIN      = 0.08   # if top-2 gallery matches are within this, it's ambiguous → new ID
REID_GALLERY_SIZE  = 5              # max number of ArcFace embeddings stored per person
REID_EMBED_MODEL   = "ArcFace"      # DeepFace model for embedding extraction
REID_DETECTOR      = "opencv"       # fast backend for embedding only
REID_VERIFY_EVERY_N    = 5          # re-check a live track's identity every N frames
REID_VERIFY_THRESHOLD  = 0.45       # if max-gallery sim drops below this the track was hijacked

# Emotion recognition
EMOTION_DETECTOR         = "skip"  # backend for DeepFace.analyze() — face already cropped
EMOTION_WINDOW           = 15      # larger window = more stable majority vote
EMOTION_EVERY_N          = 3       # run DeepFace every 3 frames (quality > quantity)
EMOTION_CONF_THRESHOLD   = 40.0   # minimum % confidence to accept (DeepFace returns 0–100)
EMOTION_SHARPNESS_MIN    = 40.0   # Laplacian variance below this → face too blurry to analyse
EMOTION_FRONTALITY_MIN   = 0.55   # width/height ratio floor — narrower → face turned sideways
FACE_CROP_PADDING        = 0.00   # 0% padding to capture ONLY the face (no hands/shoulders)
SAVE_EVERY_N_FRAMES      = 30     # save face image every 30 frames (~1 sec at 30 fps)
RECORD_EVERY_N_FRAMES    = 5      # write JSON record every 5 frames per person

# Supported emotions (Disgust is intentionally excluded)
VALID_EMOTIONS     = {"happy", "sad", "angry", "fear", "surprise", "neutral"}
EMOTION_LABEL_MAP  = {
    "happy":    "Happy",
    "sad":      "Sad",
    "angry":    "Angry",
    "fear":     "Fear",
    "surprise": "Surprise",
    "neutral":  "Neutral",
    "disgust":  "Neutral",   # remap Disgust → Neutral
    "normal":   "Neutral",
}

# Annotation colours  (BGR)
BOX_COLOR          = (0, 255, 128)       # mint green
TEXT_BG_COLOR      = (15, 15, 15)        # near-black
TEXT_COLOR         = (255, 255, 255)     # white
ID_COLOR           = (0, 215, 255)       # gold

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dirs() -> None:
    """Create the captured_faces sub-folders if they don't exist."""
    for emotion in ["Happy", "Sad", "Angry", "Fear", "Surprise", "Neutral"]:
        os.makedirs(os.path.join(FACES_DIR, emotion), exist_ok=True)


def map_emotion(raw: str) -> str:
    """
    Normalise a raw DeepFace emotion label to one of the 6 supported labels.
    Any unknown label defaults to 'Neutral'.
    """
    return EMOTION_LABEL_MAP.get(raw.lower(), "Neutral")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity in [0, 1] between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ─────────────────────────────────────────────────────────────────────────────
# FACE DETECTION  (MediaPipe BlazeFace Short Range — Tasks Vision API)
# ─────────────────────────────────────────────────────────────────────────────

# BlazeFace Short Range model — auto-downloaded on first run
BLAZE_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
BLAZE_MODEL_PATH = "blaze_face_short_range.tflite"


def _ensure_blaze_model() -> None:
    """Download the BlazeFace .tflite model if not already present."""
    if not os.path.exists(BLAZE_MODEL_PATH):
        import urllib.request
        print(f"[INFO] Downloading BlazeFace model → {BLAZE_MODEL_PATH} …")
        urllib.request.urlretrieve(BLAZE_MODEL_URL, BLAZE_MODEL_PATH)
        print(f"[INFO] BlazeFace model saved ({os.path.getsize(BLAZE_MODEL_PATH):,} bytes)")


class FaceDetector:
    """
    Wraps MediaPipe BlazeFace Short Range for per-frame face detection.
    Uses the new mp.tasks.vision.FaceDetector Tasks API (MediaPipe ≥ 0.10).
    """

    def __init__(self, min_confidence: float = BLAZE_CONFIDENCE):
        _ensure_blaze_model()
        base_options = mp.tasks.BaseOptions(model_asset_path=BLAZE_MODEL_PATH)
        options      = mp.tasks.vision.FaceDetectorOptions(
            base_options             = base_options,
            min_detection_confidence = min_confidence,
            # Lower NMS threshold = less suppression = all faces detected even when close together
            min_suppression_threshold = 0.3,
            running_mode             = mp.tasks.vision.RunningMode.IMAGE,
        )
        self._detector = mp.tasks.vision.FaceDetector.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray):
        """
        Run BlazeFace on a BGR frame.

        Returns
        -------
        list of (x1, y1, x2, y2, confidence)  — pixel coordinates, clipped to frame.
        """
        h, w  = frame_bgr.shape[:2]
        rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_img)

        detections = []
        if result and result.detections:
            for det in result.detections:
                bb   = det.bounding_box
                # MediaPipe Tasks API may return bounding box coordinates either
                # as absolute pixels or as normalized ratios in [0,1]. Detect
                # normalized values and scale them to pixel coordinates when
                # necessary so boxes align with the source frame.
                ox = bb.origin_x
                oy = bb.origin_y
                bw = bb.width
                bh = bb.height
                # If values look like normalized ratios (<= 1.0), scale them
                if 0.0 <= ox <= 1.01 and 0.0 <= oy <= 1.01 and 0.0 <= bw <= 1.01 and 0.0 <= bh <= 1.01:
                    ox = ox * w
                    bw = bw * w
                    oy = oy * h
                    bh = bh * h

                x1   = int(max(int(round(ox)), 0))
                y1   = int(max(int(round(oy)), 0))
                x2   = int(min(int(round(ox + bw)),  w - 1))
                y2   = int(min(int(round(oy + bh)), h - 1))
                conf = det.categories[0].score if det.categories else 0.5
                if x2 > x1 and y2 > y1:
                    detections.append((x1, y1, x2, y2, conf))

        return detections

    def close(self):
        self._detector.close()


# ─────────────────────────────────────────────────────────────────────────────
# FACE RE-IDENTIFICATION  (ArcFace embeddings via DeepFace)
# ─────────────────────────────────────────────────────────────────────────────

# Spatial matching constants
SPATIAL_MAX_CENTER_DIST  = 2.0    # max normalized center distance for spatial match
SPATIAL_GRACE_FRAMES     = DEEPSORT_MAX_AGE  # how long a lost track stays spatially matchable


class FaceReID:
    """
    Maintains a database of known face embeddings.
    Maps ephemeral DeepSORT track IDs to permanent, stable IDs.

    Gallery-based matching: up to REID_GALLERY_SIZE ArcFace embeddings are
    stored per person. Similarity is the MAX over the whole gallery so that
    the best-angle/lighting shot always wins, without embedding drift.

    Stability guarantees
    --------------------
    • Forward mapping  (_ds_to_perm) : DeepSORT track_id  → permanent ID
    • Reverse mapping  (_perm_to_ds) : permanent ID       → DeepSORT track_id
      Ensures no two active tracks share the same perm_id.
    • Spatial fallback : when embedding fails (face too small/blurry),
      re-entry is matched by bbox proximity to recently-lost tracks.
    • Grace-period cleanup: stale track mappings are kept alive for
      DEEPSORT_MAX_AGE frames after the track disappears, preventing
      premature re-ID when DeepSORT briefly loses a face.
    • verify_cached_track NEVER creates new IDs — only reassigns to
      known persons or keeps the current ID.
    • Post-processing merge: after the video ends, galleries are cross-
      checked and duplicate perm_ids (same person split across tracks)
      are merged back together.
    """

    def __init__(self, threshold: float = REID_SIMILARITY_THRESHOLD):
        self._threshold    = threshold
        self._next_perm_id = 1                 # next permanent ID to assign

        # ── Forward + reverse mappings ─────────────────────────────────────────
        self._ds_to_perm:     dict[int, int] = {}   # deepsort_id → perm_id
        self._perm_to_ds:     dict[int, int] = {}   # perm_id → deepsort_id (reverse)

        # ── Embedding gallery ──────────────────────────────────────────────────
        # perm_id → {"embeddings": list[np.ndarray]}
        self._known:          dict[int, dict] = {}

        # ── Verification throttle ──────────────────────────────────────────────
        self._verify_counter: dict[int, int]  = {}   # ds_id → frame counter

        # ── Periodic save / record tracking ────────────────────────────────────
        self._emotion_best:   dict[tuple, int] = {}  # (perm_id, key) → frame_num

        # ── Spatial tracking for re-entry matching ─────────────────────────────
        self._last_seen_bbox:  dict[int, tuple] = {}  # perm_id → (x1,y1,x2,y2)
        self._last_seen_frame: dict[int, int]   = {}  # perm_id → frame_num

        # ── Stale-cleanup grace period ─────────────────────────────────────────
        self._ds_last_active:  dict[int, int]   = {}  # ds_id → last active frame

    # ------------------------------------------------------------------
    def get_embedding(self, face_crop: np.ndarray) -> np.ndarray | None:
        """
        Extract ArcFace embedding from a face crop.
        Returns None on failure (too small, no face found, etc.).
        """
        if face_crop is None or face_crop.size == 0:
            return None
        h, w = face_crop.shape[:2]
        if h < 20 or w < 20:
            return None
        try:
            resized = cv2.resize(face_crop, (224, 224))
            result  = DeepFace.represent(
                img_path          = resized,
                model_name        = REID_EMBED_MODEL,
                detector_backend  = "skip",     # crop already done
                enforce_detection = False,
            )
            if result and isinstance(result, list):
                emb = np.array(result[0]["embedding"], dtype=np.float32)
                return emb
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    def _gallery_similarity(self, embedding: np.ndarray, perm_id: int) -> float:
        """
        Return the MAXIMUM cosine similarity between `embedding` and any
        vector in the stored gallery for `perm_id`.
        Returns -1.0 if the gallery is empty.
        """
        gallery = self._known.get(perm_id, {}).get("embeddings", [])
        if not gallery:
            return -1.0
        return max(cosine_similarity(embedding, g) for g in gallery)

    def _gallery_similarity_cross(self, pid_a: int, pid_b: int) -> float:
        """
        Return the MAX cosine similarity between any pair of embeddings
        from two different person galleries.  Used by post-processing merge.
        """
        gallery_a = self._known.get(pid_a, {}).get("embeddings", [])
        gallery_b = self._known.get(pid_b, {}).get("embeddings", [])
        if not gallery_a or not gallery_b:
            return -1.0
        best = -1.0
        for ea in gallery_a:
            for eb in gallery_b:
                s = cosine_similarity(ea, eb)
                if s > best:
                    best = s
        return best

    # ------------------------------------------------------------------
    def _add_to_gallery(self, perm_id: int, embedding: np.ndarray) -> None:
        """
        Append `embedding` to the person's gallery with integrity checks.

        Gallery integrity rules:
        • First embedding is always accepted (bootstraps the gallery).
        • Subsequent embeddings must pass a minimum similarity floor
          (REID_VERIFY_THRESHOLD) against the existing gallery — prevents
          a wrongly-matched embedding from corrupting the gallery.
        • When the gallery is full, the MOST REDUNDANT entry (highest
          similarity to the new embedding) is replaced, preserving
          angular diversity while keeping all entries verified.
        """
        entry   = self._known.setdefault(perm_id, {"embeddings": []})
        gallery: list = entry.setdefault("embeddings", [])

        # First embedding — always accept
        if len(gallery) == 0:
            gallery.append(embedding)
            return

        # Similarity floor: reject embeddings that are too dissimilar
        max_sim = max(cosine_similarity(embedding, g) for g in gallery)
        if max_sim < REID_VERIFY_THRESHOLD:
            return  # too dissimilar → likely a different person → protect gallery

        if len(gallery) < REID_GALLERY_SIZE:
            gallery.append(embedding)
        else:
            # Replace the MOST SIMILAR (most redundant) entry to preserve
            # angular diversity while ensuring all entries stay verified
            sims = [cosine_similarity(embedding, g) for g in gallery]
            most_redundant_idx = int(np.argmax(sims))
            gallery[most_redundant_idx] = embedding

    # ------------------------------------------------------------------
    # Spatial helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _bbox_center_distance(box1: tuple, box2: tuple) -> float:
        """
        Euclidean distance between bbox centres, normalised by the average
        bbox diagonal so the metric is resolution-independent.
        """
        cx1 = (box1[0] + box1[2]) / 2.0
        cy1 = (box1[1] + box1[3]) / 2.0
        cx2 = (box2[0] + box2[2]) / 2.0
        cy2 = (box2[1] + box2[3]) / 2.0
        dist  = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
        diag1 = ((box1[2] - box1[0]) ** 2 + (box1[3] - box1[1]) ** 2) ** 0.5
        diag2 = ((box2[2] - box2[0]) ** 2 + (box2[3] - box2[1]) ** 2) ** 0.5
        avg_diag = (diag1 + diag2) / 2.0
        return dist / avg_diag if avg_diag > 0 else float("inf")

    def _spatial_match(self, bbox: tuple, frame_num: int,
                       active_perm_ids: set[int]) -> int | None:
        """
        Find a recently-lost perm_id whose last known bbox is spatially
        close to `bbox`.  Returns None if no suitable candidate exists.
        """
        best_pid   = None
        best_score = float("inf")

        for pid, last_bbox in self._last_seen_bbox.items():
            # Skip IDs already claimed by another track this frame
            if pid in active_perm_ids:
                continue
            last_frame = self._last_seen_frame.get(pid, -9999)
            frame_gap  = frame_num - last_frame
            if frame_gap > SPATIAL_GRACE_FRAMES:
                continue   # too long ago

            dist = self._bbox_center_distance(bbox, last_bbox)
            # Combined score: prefer closer + more recent
            score = dist + frame_gap * 0.01
            if dist < SPATIAL_MAX_CENTER_DIST and score < best_score:
                best_score = score
                best_pid   = pid

        return best_pid

    def update_position(self, perm_id: int, bbox: tuple, frame_num: int) -> None:
        """Update the last-seen position for a perm_id (called every frame)."""
        self._last_seen_bbox[perm_id]  = bbox
        self._last_seen_frame[perm_id] = frame_num

    # ------------------------------------------------------------------
    # Reverse-mapping helpers
    # ------------------------------------------------------------------
    def _is_perm_id_available(self, perm_id: int, ds_id: int) -> bool:
        """
        Return True if `perm_id` can safely be assigned to `ds_id`.
        It's available if no other *active* track currently owns it.
        """
        owning_ds = self._perm_to_ds.get(perm_id)
        if owning_ds is None:
            return True
        if owning_ds == ds_id:
            return True
        # The old owner is gone — mapping is stale
        if owning_ds not in self._ds_to_perm:
            return True
        return False

    def _assign_mapping(self, ds_id: int, perm_id: int) -> None:
        """Set both forward and reverse mappings atomically."""
        # Clear old reverse entry if this ds_id previously owned a different perm_id
        old_perm = self._ds_to_perm.get(ds_id)
        if old_perm is not None and old_perm != perm_id:
            if self._perm_to_ds.get(old_perm) == ds_id:
                del self._perm_to_ds[old_perm]

        self._ds_to_perm[ds_id]   = perm_id
        self._perm_to_ds[perm_id] = ds_id

    # ------------------------------------------------------------------
    def resolve(self, ds_id: int, face_crop: np.ndarray,
                active_perm_ids: set | None = None,
                bbox: tuple | None = None,
                frame_num: int = 0) -> int:
        """
        Return the permanent ID for a given DeepSORT track ID.

        Algorithm
        ---------
        1. If ds_id already mapped → return cached perm_id directly.
        2. Extract ArcFace embedding from face_crop (skip if crop too small).
        3. Compare embedding against every person's gallery (max similarity).
        4. AMBIGUITY CHECK: if the top-2 gallery matches are within
           REID_AMBIGUITY_MARGIN of each other → match is ambiguous.
        5. If best gallery similarity ≥ threshold AND not ambiguous AND
           perm_id is available → reuse that person's ID.
        5b. SPATIAL FALLBACK: if embedding failed or was ambiguous, check if
           the bbox is spatially close to a recently-lost track.
        6. Only as a LAST RESORT, assign a fresh permanent ID.
        """
        if active_perm_ids is None:
            active_perm_ids = set()

        # ── Step 1: cached track — return stored mapping unchanged ────────────
        if ds_id in self._ds_to_perm:
            return self._ds_to_perm[ds_id]

        # ── Step 2: extract ArcFace embedding ─────────────────────────────────
        h, w = (face_crop.shape[:2] if face_crop is not None and face_crop.size > 0
                else (0, 0))
        embedding = self.get_embedding(face_crop) if h >= 50 and w >= 50 else None

        # ── Step 3: gallery matching with ambiguity detection ──────────────────
        best_perm_id  = None
        best_sim      = -1.0
        second_sim    = -1.0     # track second-best for ambiguity check

        if embedding is not None:
            for pid in self._known:
                sim = self._gallery_similarity(embedding, pid)
                if sim > best_sim:
                    second_sim   = best_sim
                    best_sim     = sim
                    best_perm_id = pid
                elif sim > second_sim:
                    second_sim   = sim

        # ── Step 4: ambiguity guard ────────────────────────────────────────────
        is_ambiguous = (second_sim >= 0 and
                        (best_sim - second_sim) < REID_AMBIGUITY_MARGIN)

        # ── Step 5: accept embedding match ─────────────────────────────────────
        perm_id = None

        if (embedding is not None
                and best_sim >= self._threshold
                and best_perm_id is not None
                and best_perm_id not in active_perm_ids
                and not is_ambiguous
                and self._is_perm_id_available(best_perm_id, ds_id)):
            perm_id = best_perm_id
            if best_sim >= REID_GALLERY_ENRICH_MIN:
                self._add_to_gallery(perm_id, embedding)

        # ── Step 5b: spatial fallback when embedding failed or ambiguous ───────
        if perm_id is None and bbox is not None:
            spatial_pid = self._spatial_match(bbox, frame_num, active_perm_ids)
            if spatial_pid is not None and self._is_perm_id_available(spatial_pid, ds_id):
                if embedding is not None:
                    # Have embedding → verify spatial candidate loosely
                    sim_check = self._gallery_similarity(embedding, spatial_pid)
                    if sim_check >= REID_VERIFY_THRESHOLD:
                        perm_id = spatial_pid
                        if sim_check >= REID_GALLERY_ENRICH_MIN:
                            self._add_to_gallery(perm_id, embedding)
                else:
                    # No embedding — trust spatial proximity if recently lost
                    last_frame = self._last_seen_frame.get(spatial_pid, -9999)
                    if (frame_num - last_frame) <= SPATIAL_GRACE_FRAMES:
                        perm_id = spatial_pid

        # ── Step 6: create new ID only as LAST RESORT ──────────────────────────
        if perm_id is None:
            perm_id = self._next_perm_id
            self._next_perm_id += 1
            if embedding is not None:
                self._add_to_gallery(perm_id, embedding)

        self._assign_mapping(ds_id, perm_id)
        return perm_id

    # ------------------------------------------------------------------
    def update_embedding(self, perm_id: int, embedding: np.ndarray) -> None:
        """Add a fresh embedding to this person's gallery (called periodically).
        Only enriches if the embedding is highly similar to existing gallery."""
        if embedding is None:
            return
        sim = self._gallery_similarity(embedding, perm_id)
        # Only add if no gallery yet (first embedding) or if highly similar
        if sim < 0 or sim >= REID_GALLERY_ENRICH_MIN:
            self._add_to_gallery(perm_id, embedding)

    # ------------------------------------------------------------------
    def cleanup_stale(self, active_ds_ids: set[int],
                      current_frame_num: int) -> None:
        """
        Remove _ds_to_perm entries for DeepSORT tracks that have been dead
        for longer than DEEPSORT_MAX_AGE frames.

        KEY DIFFERENCE from the old version: instead of purging ALL inactive
        tracks immediately, we keep the mapping alive for a grace period so
        that a briefly-occluded face can rejoin its track without triggering
        a fresh (potentially wrong) ReID.
        """
        # Record last-active frame for every currently-alive track
        for ds_id in active_ds_ids:
            self._ds_last_active[ds_id] = current_frame_num

        # Purge only tracks that have been dead long enough
        stale_ids = []
        for ds_id in list(self._ds_to_perm.keys()):
            if ds_id not in active_ds_ids:
                last_active = self._ds_last_active.get(ds_id, 0)
                if (current_frame_num - last_active) > DEEPSORT_MAX_AGE:
                    stale_ids.append(ds_id)

        for ds_id in stale_ids:
            perm_id = self._ds_to_perm.pop(ds_id, None)
            # Clear reverse mapping only if this ds_id still owns it
            if perm_id is not None and self._perm_to_ds.get(perm_id) == ds_id:
                del self._perm_to_ds[perm_id]
            self._verify_counter.pop(ds_id, None)
            self._ds_last_active.pop(ds_id, None)

    # ------------------------------------------------------------------
    def verify_cached_track(self, ds_id: int, face_crop: np.ndarray,
                             active_perm_ids: set[int]) -> int | None:
        """
        Periodically re-verify that a cached ds_id still belongs to the
        same person stored in its perm_id's gallery.

        Three zones of similarity:
          sim ≥ GALLERY_ENRICH_MIN (0.85) : same person → enrich gallery
          sim ≥ VERIFY_THRESHOLD   (0.45) : plausible → keep ID, DON'T enrich
          sim <  VERIFY_THRESHOLD  (0.45) : different → try reassign to KNOWN person

        CRITICAL: this method NEVER creates a new perm_id.  If no existing
        person matches, the current ID is kept (benefit of the doubt).
        This prevents the verify step from fragmenting a single person
        into multiple IDs due to noisy embeddings.

        Returns new perm_id if a correction was made, else None.
        """
        if ds_id not in self._ds_to_perm:
            return None

        cnt = self._verify_counter.get(ds_id, 0) + 1
        self._verify_counter[ds_id] = cnt
        if cnt % REID_VERIFY_EVERY_N != 0:
            return None   # not time to check yet

        h, w = (face_crop.shape[:2]
                if face_crop is not None and face_crop.size > 0 else (0, 0))
        if h < 50 or w < 50:
            return None   # crop too small for a reliable check

        embedding = self.get_embedding(face_crop)
        if embedding is None:
            return None

        current_perm_id = self._ds_to_perm[ds_id]
        sim = self._gallery_similarity(embedding, current_perm_id)

        if sim >= REID_GALLERY_ENRICH_MIN:
            # Very high confidence — same person, enrich gallery
            self._add_to_gallery(current_perm_id, embedding)
            return None

        if sim >= REID_VERIFY_THRESHOLD:
            # Plausible same person — keep ID but do NOT enrich gallery
            return None

        # ── Possible hijack — find best EXISTING match (never create new) ─────
        best_perm_id = None
        best_sim     = -1.0
        for pid in self._known:
            if pid == current_perm_id:
                continue
            s = self._gallery_similarity(embedding, pid)
            if s > best_sim:
                best_sim     = s
                best_perm_id = pid

        # Only reassign if we have a HIGH-confidence match to a KNOWN person
        # AND that person's ID is available (not held by another active track)
        if (best_sim >= self._threshold
                and best_perm_id is not None
                and best_perm_id not in active_perm_ids
                and self._is_perm_id_available(best_perm_id, ds_id)):
            self._assign_mapping(ds_id, best_perm_id)
            self._add_to_gallery(best_perm_id, embedding)
            return best_perm_id

        # No clean match → keep current ID (NEVER create a new ID here!)
        return None

    # ------------------------------------------------------------------
    # Post-processing: merge duplicate perm_ids
    # ------------------------------------------------------------------
    def merge_duplicate_ids(self) -> dict[int, int]:
        """
        After the video ends, cross-check all person galleries.
        If two perm_ids have gallery similarity ≥ REID_GALLERY_ENRICH_MIN,
        they are almost certainly the same person split across separate
        tracks.  Return a mapping {old_id → canonical_id} for merging.
        """
        merge_map: dict[int, int] = {}  # old_id → canonical_id
        perm_ids = sorted(self._known.keys())

        for i, pid_a in enumerate(perm_ids):
            # If pid_a is already being merged into someone else, skip
            canonical_a = pid_a
            while canonical_a in merge_map:
                canonical_a = merge_map[canonical_a]

            for pid_b in perm_ids[i + 1:]:
                if pid_b in merge_map:
                    continue
                sim = self._gallery_similarity_cross(canonical_a, pid_b)
                if sim >= REID_GALLERY_ENRICH_MIN:
                    # Same person — merge pid_b → canonical_a (lower ID wins)
                    merge_map[pid_b] = canonical_a

        return merge_map

    # ------------------------------------------------------------------
    # Periodic Saving Tracking (per Person + Emotion)
    # ------------------------------------------------------------------
    def get_last_saved_frame(self, perm_id: int, emotion: str) -> int:
        """Return the frame number when the last image was saved for this person and emotion."""
        return self._emotion_best.get((perm_id, emotion), -9999)

    def set_last_saved_frame(self, perm_id: int, emotion: str, frame_num: int) -> None:
        """Store the frame number of the last saved crop for this emotion."""
        self._emotion_best[(perm_id, emotion)] = frame_num

    def get_last_recorded_frame(self, perm_id: int) -> int:
        """Return last frame a JSON record was written for this person."""
        return self._emotion_best.get((perm_id, "__record__"), -9999)

    def set_last_recorded_frame(self, perm_id: int, frame_num: int) -> None:
        self._emotion_best[(perm_id, "__record__")] = frame_num


# ─────────────────────────────────────────────────────────────────────────────
# EMOTION SMOOTHER
# ─────────────────────────────────────────────────────────────────────────────

class EmotionSmoother:
    """
    Per-ID rolling window that returns a CONFIDENCE-WEIGHTED majority-vote
    emotion to prevent flickering labels.

    Each entry in the history is (emotion_str, confidence_float) where
    confidence is DeepFace's raw score for that emotion (0–100).
    The winning label is the one whose summed confidence is highest,
    so a single 90 %-confident 'Neutral' outweighs six 5 %-confident 'Fear'
    entries — eliminating spurious flickers from low-quality frames.
    """

    def __init__(self, window: int = EMOTION_WINDOW):
        self._window  = window
        # perm_id → deque of (emotion_str, confidence)
        self._history: dict[int, deque] = {}
        self._counter: dict[int, int]   = {}   # frames since last DeepFace call

    def add(self, perm_id: int, raw_emotion: str, confidence: float = 50.0) -> None:
        if perm_id not in self._history:
            self._history[perm_id] = deque(maxlen=self._window)
        self._history[perm_id].append((map_emotion(raw_emotion), confidence))

    def get(self, perm_id: int) -> str:
        hist = self._history.get(perm_id)
        if not hist:
            return "Neutral"
        # Weighted vote: accumulate confidence scores per emotion label
        scores: dict[str, float] = {}
        for (emotion, conf) in hist:
            scores[emotion] = scores.get(emotion, 0.0) + conf
        return max(scores, key=scores.__getitem__)

    def should_run(self, perm_id: int) -> bool:
        """Throttle DeepFace calls — run every EMOTION_EVERY_N frames per ID."""
        cnt = self._counter.get(perm_id, 0)
        self._counter[perm_id] = (cnt + 1) % EMOTION_EVERY_N
        return cnt == 0


# ─────────────────────────────────────────────────────────────────────────────
# EMOTION RECOGNITION  (DeepFace)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """
    Enhance face crop before passing to DeepFace:
      - Convert to LAB colour space
      - Apply CLAHE to the L channel (boosts contrast / handles low light)
      - Convert back to BGR
    """
    lab  = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l_eq  = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# Minimum fraction of skin-coloured pixels required to attempt emotion analysis.
# If a face is covered by hair, a hat, or turned sideways, the skin ratio
# will drop below this and we return "Neutral" without calling DeepFace.
SKIN_RATIO_THRESHOLD = 0.12   # 12 %


def is_face_obstructed(face_bgr: np.ndarray) -> bool:
    """
    Disabled to ensure expression detection does not erroneously filter faces.
    """
    return False


def _face_sharpness(face_bgr: np.ndarray) -> float:
    """Return Laplacian variance — a proxy for image sharpness.
    Values below EMOTION_SHARPNESS_MIN indicate a blurry / low-quality crop."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _face_is_frontal(face_bgr: np.ndarray) -> bool:
    """Return True if the face looks roughly frontal (not turned sideways).
    A very narrow face relative to its height means the person is in profile."""
    h, w = face_bgr.shape[:2]
    if h == 0:
        return False
    return (w / h) >= EMOTION_FRONTALITY_MIN


def analyze_emotion(face_crop: np.ndarray,
                    prev_emotion: str = "Neutral") -> tuple[str, float]:
    """
    Run DeepFace emotion analysis on a face crop.

    Quality gates applied BEFORE calling DeepFace (all fall back to prev_emotion):
      ─ Minimum size          : face must be ≥ 20×20 px
      ─ Sharpness gate        : Laplacian variance ≥ EMOTION_SHARPNESS_MIN
      ─ Frontality gate       : width/height ≥ EMOTION_FRONTALITY_MIN

    Processing:
      ─ CLAHE contrast enhancement
      ─ 256×256 resize for better feature resolution
      ─ Confidence threshold  : only accept dominant emotion if score ≥ EMOTION_CONF_THRESHOLD

    Returns
    -------
    (emotion_str, confidence_float)
      emotion_str  : normalised label (never 'Disgust')
      confidence   : DeepFace score 0–100 for the accepted emotion;
                     50.0 when falling back to prev_emotion (neutral weight)
    """
    if face_crop is None or face_crop.size == 0:
        return prev_emotion, 50.0
    h, w = face_crop.shape[:2]
    if h < 20 or w < 20:
        return prev_emotion, 50.0

    # ── Sharpness gate ──────────────────────────────────────────────────────
    # Blurry frames produce random emotion outputs from DeepFace. Skip them.
    if _face_sharpness(face_crop) < EMOTION_SHARPNESS_MIN:
        return prev_emotion, 50.0

    # ── Frontality gate ────────────────────────────────────────────────────
    # Profile / turned faces give unreliable emotion predictions. Skip them.
    if not _face_is_frontal(face_crop):
        return prev_emotion, 50.0

    try:
        # Preprocess: CLAHE contrast boost
        enhanced = preprocess_face(face_crop)
        # Resize to 256×256 for better feature resolution
        resized = cv2.resize(enhanced, (256, 256))
        result  = DeepFace.analyze(
            img_path          = resized,
            actions           = ["emotion"],
            detector_backend  = EMOTION_DETECTOR,
            enforce_detection = False,
            silent            = True,
        )
        if isinstance(result, list):
            result = result[0]

        dominant   = result.get("dominant_emotion", "neutral")
        # Guard against empty / whitespace emotion string from DeepFace
        if not dominant or not dominant.strip():
            return prev_emotion, 50.0

        scores     = result.get("emotion", {})
        confidence = scores.get(dominant, 0.0)

        # Reject low-confidence predictions — keep previous stable label
        if confidence < EMOTION_CONF_THRESHOLD:
            return prev_emotion, 50.0

        return map_emotion(dominant), float(confidence)

    except Exception:
        return prev_emotion, 50.0


# ─────────────────────────────────────────────────────────────────────────────
# FACE SAVING
# ─────────────────────────────────────────────────────────────────────────────

def save_face_periodically(
    frame_bgr:  np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    perm_id:    int,
    emotion:    str,
    frame_num:  int,
    reid:       FaceReID,
) -> str:
    """
    Save the face crop when an active emotion is detected.
    Neutral is only saved if the person has never been captured before.
    Returns relative path of saved image or empty string.
    """
    has_saved_any = any(k[0] == perm_id and k[1] != "__record__" for k in reid._emotion_best.keys())
    
    if emotion == "Neutral" and has_saved_any:
        return ""

    last_saved = reid.get_last_saved_frame(perm_id, emotion)
    if (frame_num - last_saved) < SAVE_EVERY_N_FRAMES:
        return ""

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return ""

    emotion_dir = os.path.join(FACES_DIR, emotion)
    os.makedirs(emotion_dir, exist_ok=True)
    filename = f"person{perm_id}_{frame_num}.jpg"
    abs_path = os.path.join(emotion_dir, filename)

    cv2.imwrite(abs_path, crop)
    reid.set_last_saved_frame(perm_id, emotion, frame_num)
    # Return relative path from run_dir so dashboard can locate it
    return os.path.join("captured_faces", emotion, filename).replace("\\", "/")


# ─────────────────────────────────────────────────────────────────────────────
# FRAME ANNOTATION
# ─────────────────────────────────────────────────────────────────────────────

def annotate_frame(
    frame:    np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    perm_id:  int,
    emotion:  str,
    ts:       str,
) -> None:
    """
    Draw a bounding box and a professional label panel on the frame (in-place).
    """
    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

    label_id     = f"ID:{perm_id}"
    label_emo    = emotion
    label_ts     = ts

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness  = 1
    pad        = 4

    (tw_id,  th_id),     _ = cv2.getTextSize(label_id,     font, font_scale, thickness)
    (tw_emo, th_emo),    _ = cv2.getTextSize(label_emo,    font, font_scale, thickness)
    (tw_ts,  th_ts),     _ = cv2.getTextSize(label_ts,     font, font_scale, thickness)

    panel_w  = max(tw_id, tw_emo, tw_ts) + pad * 2
    line_h   = max(th_id, th_emo, th_ts) + pad * 2
    panel_h  = line_h * 3

    # Position panel above the box; clamp to frame top
    px1 = x1
    py2 = y1
    py1 = max(0, py2 - panel_h)
    px2 = min(frame.shape[1] - 1, px1 + panel_w)

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (px1, py1), (px2, py2), TEXT_BG_COLOR, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Text lines
    y_cursor = py1 + line_h - pad
    cv2.putText(frame, label_id,      (px1 + pad, y_cursor), font, font_scale, ID_COLOR,    thickness, cv2.LINE_AA)
    y_cursor += line_h
    cv2.putText(frame, label_emo,    (px1 + pad, y_cursor), font, font_scale, TEXT_COLOR,  thickness, cv2.LINE_AA)
    y_cursor += line_h
    cv2.putText(frame, label_ts,     (px1 + pad, y_cursor), font, font_scale, (180,180,180), thickness, cv2.LINE_AA)


def draw_overlay_info(frame: np.ndarray, frame_num: int, total: int, fps: float) -> None:
    """Draw a small HUD in the top-right corner with frame / FPS info."""
    h, w = frame.shape[:2]
    info  = f"Frame {frame_num}/{total}  |  {fps:.1f} FPS"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(info, font, 0.45, 1)
    x = w - tw - 10
    y = th + 8
    cv2.rectangle(frame, (x - 4, 4), (w - 6, y + 6), (20, 20, 20), -1)
    cv2.putText(frame, info, (x, y), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main(run_dir: str = ".") -> None:
    """
    Full pipeline:
      1. Open input video → read FPS, resolution, frame count.
      2. For every frame:
         a. Detect faces with MediaPipe BlazeFace.
         b. Feed detections to DeepSORT.
         c. Resolve ephemeral DeepSORT IDs → permanent IDs via Face ReID.
         d. Run DeepFace emotion (throttled + smoothed).
         e. Save best face crop to disk.
         f. Annotate frame.
         g. Write to output video.
         h. Display live with cv2.imshow.
      3. Save result.json.
    """
    global INPUT_VIDEO, OUTPUT_VIDEO, RESULT_JSON, FACES_DIR
    INPUT_VIDEO = os.path.join(run_dir, "input_video.mp4")
    OUTPUT_VIDEO = os.path.join(run_dir, "output_video.mp4")
    RESULT_JSON = os.path.join(run_dir, "result.json")
    FACES_DIR = os.path.join(run_dir, "captured_faces")
    ensure_dirs()

    # ── Open input video ──────────────────────────────────────────────────────
    if not os.path.exists(INPUT_VIDEO):
        print(f"[ERROR] Input video not found: '{INPUT_VIDEO}'")
        print("        Ensure you provided a valid --video path.")
        return

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: '{INPUT_VIDEO}'")
        return

    fps         = round(cap.get(cv2.CAP_PROP_FPS) or 30.0, 2)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Input  : {INPUT_VIDEO}  (@ {fps:.2f} FPS,  {total_frames} frames)")

    # We'll read the first frame immediately and create the VideoWriter
    # from the actual frame shape. This ensures the output video's
    # resolution (and aspect ratio) exactly matches the input frame data.
    ret, first_frame = cap.read()
    if not ret:
        print(f"[ERROR] Cannot read first frame from: '{INPUT_VIDEO}'")
        return
    # Use the actual frame dimensions (width, height)
    height, width = first_frame.shape[:2]
    # Many video encoders require even dimensions — crop a single row/col
    # if necessary so the writer doesn't silently change resolution.
    crop_right = 0
    crop_bottom = 0
    if width % 2 != 0:
        crop_right = 1
        width -= 1
    if height % 2 != 0:
        crop_bottom = 1
        height -= 1
    if crop_right or crop_bottom:
        first_frame = first_frame[0:height, 0:width]
    print(f"[INFO] Input  : {INPUT_VIDEO}  ({width}×{height} @ {fps:.2f} FPS,  {total_frames} frames)")

    # ── Video writer ──────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    # ── Initialise pipeline components ───────────────────────────────────────
    detector = FaceDetector(min_confidence=BLAZE_CONFIDENCE)

    tracker  = DeepSort(
        max_age             = DEEPSORT_MAX_AGE,
        n_init              = DEEPSORT_N_INIT,
        max_iou_distance    = DEEPSORT_MAX_IOU,
        embedder            = "mobilenet",   # built-in embedder for IoU motion tracking
        half                = False,
        embedder_gpu        = False,         # CPU-only
        bgr                 = True,
    )

    reid     = FaceReID(threshold=REID_SIMILARITY_THRESHOLD)
    smoother = EmotionSmoother(window=EMOTION_WINDOW)

    json_records: list[dict] = []
    wait_ms = max(1, int(1000 / fps))       # correct live-playback delay

    print("[INFO] Processing — press Q to quit early …")
    print("[INFO] (Warming up AI models, please wait...)")
    
    # Warm up DeepFace so the first frame doesn't freeze when lazy-loading TensorFlow
    dummy_face = np.zeros((224, 224, 3), dtype=np.uint8)
    _ = reid.get_embedding(dummy_face)
    _ = analyze_emotion(dummy_face)

    frame_num  = 0
    start_time = time.time()

    # ── Show first frame immediately so window pops up instantly ──────────────
    cv2.imshow("Face Detection & Expression Recognition", first_frame)
    cv2.waitKey(1)
    # We will process the frame we already read above
    current_frame = first_frame



    # ─────────────────────────────────────────────────────────────────────────
    # FRAME LOOP
    # ─────────────────────────────────────────────────────────────────────────
    while current_frame is not None:
        loop_start = time.time()
        frame = current_frame

        frame_num += 1
        ts_now     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── 1. Detect faces ───────────────────────────────────────────────────
        raw_dets = detector.detect(frame)

        # ── 2. Format for DeepSORT: [[x1,y1,w,h], conf, class_id] ────────────
        ds_input = []
        for (x1, y1, x2, y2, conf) in raw_dets:
            ds_input.append(([x1, y1, x2 - x1, y2 - y1], conf, 0))

        # ── 3. Update tracker (pass full frame for built-in mobilenet embedder) ─
        tracks = tracker.update_tracks(ds_input, frame=frame)

        # ── 4. Process ALL tracks (confirmed + tentative) ─────────────────────
        # Tentative tracks = faces seen for first time — we still draw their box.
        # active_perm_ids tracks which permanent IDs are already in use THIS frame
        # to prevent two different people sharing the same ID.
        #
        # KEY FIX: purge stale DeepSORT track mappings BEFORE processing.
        # DeepSORT recycles track_ids after a track is lost. Without this step,
        # a new person that gets an old (recycled) track_id instantly inherits
        # the dead person's perm_id from the _ds_to_perm cache — bypassing
        # all embedding checks entirely.
        active_ds_ids: set[int] = {
            t.track_id for t in tracks
            if t.is_confirmed() or t.is_tentative()
        }
        reid.cleanup_stale(active_ds_ids, frame_num)

        active_perm_ids: set[int] = set()

        for track in tracks:
            # Skip tracks with no valid bounding box
            if not track.is_confirmed() and not track.is_tentative():
                continue

            ds_id   = track.track_id
            ltrb    = track.to_ltrb()
            x1 = int(max(ltrb[0], 0))
            y1 = int(max(ltrb[1], 0))
            x2 = int(min(ltrb[2], width  - 1))
            y2 = int(min(ltrb[3], height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            # ── 5a. Check if face is currently visible (not backwards/occluded) ──
            # time_since_update == 0 means BlazeFace gave us a fresh detection
            # time_since_update  > 0 means Kalman-filter prediction — face not visible
            face_is_visible = (track.time_since_update == 0)

            # ── 5b. Extract padded crop for better emotion analysis ───────────
            pad_x = int((x2 - x1) * FACE_CROP_PADDING)
            pad_y = int((y2 - y1) * FACE_CROP_PADDING)
            px1 = max(0, x1 - pad_x)
            py1 = max(0, y1 - pad_y)
            px2 = min(width  - 1, x2 + pad_x)
            py2 = min(height - 1, y2 + pad_y)
            face_crop         = frame[py1:py2, px1:px2]   # padded (for emotion)
            face_crop_tight   = frame[y1:y2, x1:x2]       # tight (for ReID)

            # ── 5. Resolve permanent ID via ReID ─────────────────────────────
            perm_id = reid.resolve(ds_id, face_crop_tight, active_perm_ids,
                                   bbox=(x1, y1, x2, y2), frame_num=frame_num)
            active_perm_ids.add(perm_id)  # register so next track in frame can't reuse it
            reid.update_position(perm_id, (x1, y1, x2, y2), frame_num)

            # ── 5c. Periodic identity verification (catches track hijacking) ──
            # Every REID_VERIFY_EVERY_N frames on a visible face, re-check the
            # embedding against the stored gallery. If similarity is too low
            # (clearly a different person), reassign to the correct perm_id.
            if face_is_visible:
                corrected = reid.verify_cached_track(ds_id, face_crop_tight, active_perm_ids)
                if corrected is not None:
                    active_perm_ids.discard(perm_id)   # release the wrong ID
                    perm_id = corrected
                    active_perm_ids.add(perm_id)       # claim the correct ID

            # ── 6. Emotion recognition (throttled + smoothed) ─────────────────
            if not face_is_visible:
                # Face turned backwards or fully occluded — add neutral at low
                # weight so it doesn't override confident recent history
                smoother.add(perm_id, "Neutral", confidence=10.0)
            else:
                prev_emotion = smoother.get(perm_id)
                if smoother.should_run(perm_id):
                    raw_emotion, conf = analyze_emotion(face_crop, prev_emotion)
                    smoother.add(perm_id, raw_emotion, confidence=conf)

            emotion = smoother.get(perm_id)

            # ── 7. Save face crop periodically (every 2s) ─────────────────────
            face_path = save_face_periodically(frame, x1, y1, x2, y2, perm_id, emotion, frame_num, reid)

            # ── 8. Build JSON record (only every N frames per person) ─────────
            last_rec = reid.get_last_recorded_frame(perm_id)
            if (frame_num - last_rec) >= RECORD_EVERY_N_FRAMES:
                json_records.append({
                    "track_id":     perm_id,

                    "timestamp":    ts_now,
                    "frame_number": frame_num,
                    "expression":   emotion,
                    "face_image":   face_path,
                })
                reid.set_last_recorded_frame(perm_id, frame_num)

            # ── 9. Annotate frame — only when face is actually detected ─────
            # Fix 3: skip drawing when face_is_visible is False (Kalman prediction)
            # so only one border appears per real detected face.
            if face_is_visible:
                annotate_frame(frame, x1, y1, x2, y2, perm_id, emotion, ts_now)

        # ── HUD overlay ───────────────────────────────────────────────────────
        draw_overlay_info(frame, frame_num, total_frames, fps)

        # ── Write output frame ────────────────────────────────────────────────
        # Ensure the frame exactly matches the writer's expected size.
        fh, fw = frame.shape[:2]
        if (fh, fw) != (height, width):
            # If frame is larger, crop; if smaller, resize (rare).
            if fh >= height and fw >= width:
                frame = frame[0:height, 0:width]
            else:
                frame = cv2.resize(frame, (width, height))
        writer.write(frame)

        # ── Live display ──────────────────────────────────────────────────────
        cv2.imshow("Face Detection & Expression Recognition", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            print("[INFO] User pressed Q — exiting early …")
            break

        # ── Advance to next frame ─────────────────────────────────────────────
        ret, current_frame = cap.read()
        if not ret:
            current_frame = None

        # Progress log every 50 frames
        if frame_num % 50 == 0:
            elapsed  = time.time() - start_time
            est_total = (elapsed / frame_num) * total_frames if frame_num else 0
            remaining = max(0, est_total - elapsed)
            print(
                f"[INFO] Frame {frame_num}/{total_frames} "
                f"| Elapsed {elapsed:.0f}s | ETA {remaining:.0f}s"
            )

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    detector.close()

    elapsed = time.time() - start_time
    print(f"[INFO] Processed {frame_num} frames in {elapsed:.1f}s")
    print(f"[INFO] Output video  : {OUTPUT_VIDEO}")

    # ── Post-processing: merge duplicate IDs ───────────────────────────────────
    merge_map = reid.merge_duplicate_ids()
    if merge_map:
        merged_pairs = [f"{old}→{new}" for old, new in sorted(merge_map.items())]
        print(f"[INFO] Merging duplicate IDs: {', '.join(merged_pairs)}")
        # Re-map all JSON records and rebuild guest IDs
        for rec in json_records:
            old_id = rec["track_id"]
            while old_id in merge_map:
                old_id = merge_map[old_id]
            rec["track_id"] = old_id


    # ── Save result.json ──────────────────────────────────────────────────────
    # Deduplicate: keep only the last record per (track_id, frame_number) pair
    seen  = {}
    for rec in json_records:
        key = (rec["track_id"], rec["frame_number"])
        seen[key] = rec
    final_records = list(seen.values())

    # Count unique persons for summary
    unique_persons = len(set(r["track_id"] for r in final_records)) if final_records else 0

    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_records, f, indent=4, ensure_ascii=False)

    print(f"[INFO] Result JSON   : {RESULT_JSON}  ({len(final_records)} records, {unique_persons} unique persons)")
    print("[DONE] Pipeline complete.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import glob
    import shutil
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    args = parser.parse_args()
    
    outputs_dir = "Outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    runs = [r for r in glob.glob(os.path.join(outputs_dir, "run*")) if os.path.isdir(r)]
    next_num = 1
    if runs:
        runs.sort(reverse=True)
        try:
            next_num = int(os.path.basename(runs[0]).replace("run", "")) + 1
        except:
            pass
    new_run_dir = os.path.join(outputs_dir, f"run{next_num:03d}")
    os.makedirs(new_run_dir, exist_ok=True)
    
    dest_video = os.path.join(new_run_dir, "input_video.mp4")
    shutil.copy(args.video, dest_video)
    
    print(f"\n[INFO] Starting processing in directory: {new_run_dir}")
    main(new_run_dir)

