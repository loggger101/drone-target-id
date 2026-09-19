#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
realtime_yolo_plus_classifier.py — YOLO + your COARSE-only classifier + simple tracking + masks

What this does
--------------
• Runs a YOLO SEGMENTATION detector to find multiple objects per frame
• Uses instance masks (not just boxes) to color the actual object shape
• Runs YOUR trained Keras classifier on each detected crop, but ONLY uses the COARSE head:
   - Coarse classes come from label_map.json["coarse_classes"], e.g.:
         ["invalid_background", "invalid_nontarget", "valid"]
   - "valid"             -> green mask + label
   - "invalid_nontarget" -> red mask + label
   - "invalid_background"-> IGNORE (no mask / no label drawn)
• Adds a simple IoU-based tracker:
   - Assigns persistent track IDs across frames where boxes overlap
   - Drops tracks that disappear for a while
• Overlays per object:
   - Track ID + coarse label + coarse prob   (NO YOLO class names shown)
• Works with webcam or a video file; can also save annotated video
• Supports tiling for better small-object detection
• For video files: tries to keep REAL-TIME playback by skipping frames if processing is too slow.
  - Disable with --no-skip
  - Limit how many frames are skipped per step with --max-skip

Extra in this version
---------------------
• Real-time path:
    - Same as before for display
    - Optional --save path (manual) still writes a real-time overlay (may be time-compressed if skipping)
• Offline path (second pass, after real-time preview, for video files only, and only if autosave is enabled):
    - Saves output/original.mp4       (no skips, original frames)
    - Saves output/overlay_coarse.mp4 (no skips, coarse masks + coarse labels)
"""


### Will run a realtime overlay of the video whihc can be watch BUT saves the original used video and a FULL runthrough of the video (wont be realtime proccessing)(hyperpereameters are optimized for realtime and as such FULL proccessing is far from optimal or hte best that can be done)###


import os
import sys
import time
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

# YOLO (Ultralytics)
try:
    from ultralytics import YOLO
except Exception as e:
    print("[ERROR] Ultralytics YOLO not installed. Run:  pip install ultralytics", file=sys.stderr)
    raise

# ------------------------
# CLI
# ------------------------
def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Classifier (your model)
    ap.add_argument("--model", type=str,
                    default="data_kagglehub_unified/output/best_model.h5",
                    # originally C:/Users/Owner/best_model.h5
                    help="Path to your Keras classifier (.h5/.keras) with a coarse head")
    ap.add_argument("--label-map", type=str,
                    default="data_kagglehub_unified/label_map.json",
                    # originally C:/Users/Owner/metadata.jsonl
                    help="Path to label_map.json (must contain 'coarse_classes')")

    # YOLO detector — now using a *segmentation* model by default
    ap.add_argument("--yolo-weights", type=str, default="yolov8n-seg.pt",
                    help="YOLO *segmentation* weights (e.g., yolov8n-seg.pt, yolov8s-seg.pt, or a custom .pt)")
    ap.add_argument("--yolo-imgsz", type=int, default=1216,
                    help="YOLO inference size (short side). Increase for small objects.")
    # More willing to keep detections
    ap.add_argument("--yolo-conf", type=float, default=0.3,
                    help="YOLO confidence threshold (lower = more boxes(and slower))")
    ap.add_argument("--yolo-iou", type=float, default=0.75,
                    help="YOLO NMS IoU threshold (higher = keep more overlapping boxes)")
    ap.add_argument("--yolo-max-det", type=int, default=50000,
                    help="Maximum number of detections per image")
    ap.add_argument("--yolo-classes", type=str, default=None,
                    help="Restrict YOLO classes by names (comma-separated). Default uses all.")

    # Tiled detection (for lots of small parts)
    ap.add_argument("--tiles", type=int, nargs=2, default=[3, 3], metavar=("ROWS", "COLS"),
                    help="Tile grid (rows cols). >1 enables tiling (helps small/packed objects).")
    ap.add_argument("--tile-overlap", type=float, default=0.5,
                    help="Fractional overlap between tiles (0–0.5). Higher = more redundancy.")

    # Classifier input size
    ap.add_argument("--clf-img-size", type=int, nargs=2, default=[256, 256], metavar=("H","W"),
                    help="Input size for your classifier crops")

    # Input / output
    ap.add_argument("--source", type=str,
                    default="0",
                    # originally a local file, e.g. C:/Users/Owner/OneDrive/Documents/DLVIDEOS/v3.mp4
                    help='Webcam index like "0" (string) or video path')
    ap.add_argument("--save", type=str, default=None, help="Optional path to save annotated video")
    ap.add_argument("--no-display", action="store_true", help="Disable window display (headless)")

    # Autosave of multiple videos into output/ folder
    ap.add_argument("--no-autosave-output", action="store_true",
                    help="Disable autosaving videos to 'output' folder (original/coarse)")

    # Overlay
    ap.add_argument("--font-scale", type=float, default=0.5, help="Overlay font scale")
    ap.add_argument("--thickness", type=int, default=1, help="Overlay line thickness")

    # Real-time control
    ap.add_argument("--no-skip", action="store_true",
                    help="Disable real-time frame skipping (process every frame, can be slower than real time)")
    ap.add_argument("--max-skip", type=int, default=50,
                    help="Maximum frames to skip per processed frame in real-time mode (video files only). "
                         "Set 0 to never skip more than the current frame (similar to --no-skip).")

    # Tracking control (simple IoU-based)
    ap.add_argument("--track-iou", type=float, default=0.5,
                    help="IoU threshold for matching detections to existing tracks")
    ap.add_argument("--track-max-age", type=int, default=90,
                    help="Maximum number of frames to keep a track without seeing it again")

    args, _ = ap.parse_known_args()  # <-- swallow unknown args like -f kernel.json
    return args

# ------------------------
# Helpers
# ------------------------
def load_coarse_classes(label_map_path: str):
    """
    Load list of coarse classes from label_map.json["coarse_classes"].
    Expected to include at least: "invalid_background", "invalid_nontarget", "valid".
    """
    p = Path(label_map_path)
    if not p.exists():
        print(f"[WARN] label_map not found at {p}, falling back to default coarse classes.")
        return ["invalid_background", "invalid_nontarget", "valid"]
    try:
        with open(p, "r", encoding="utf-8") as f:
            m = json.load(f)
        if isinstance(m, dict) and "coarse_classes" in m and isinstance(m["coarse_classes"], list):
            return m["coarse_classes"]
    except Exception as e:
        print(f"[WARN] Failed to parse coarse_classes from label_map: {e}")
    return ["invalid_background", "invalid_nontarget", "valid"]

def preprocess_for_classifier(bgr: np.ndarray, size_hw: tuple[int,int]) -> np.ndarray:
    """Resize BGR -> RGB, scale to [0,1], add batch dim."""
    h, w = size_hw
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    return np.expand_dims(x, axis=0)

def draw_label_box(img, text, topleft, font_scale=0.6, thickness=2,
                   fg=(255,255,255), bg=(0,0,0)):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = int(topleft[0]), int(topleft[1])
    # Draw filled rectangle behind text
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + base + 6), bg, -1)
    # Put text
    cv2.putText(img, text, (x + 3, y + base), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, fg, thickness, cv2.LINE_AA)

def clip_xyxy(xyxy, W, H):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1 = max(0, min(W-1, x1))
    y1 = max(0, min(H-1, y1))
    x2 = max(0, min(W-1, x2))
    y2 = max(0, min(H-1, y2))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2

# ---- NMS for merged tile detections ----
def iou_box(box1, box2):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    inter_x1 = max(x1, x1b)
    inter_y1 = max(y1, y1b)
    inter_x2 = min(x2, x2b)
    inter_y2 = min(y2, y2b)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area1 = max(0, x2 - x1) * max(0, y2 - y1)
    area2 = max(0, x2b - x1b) * max(0, y2b - y1b)
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union

def nms_detections(dets, iou_thres=0.6):
    """
    Simple class-aware NMS over a list of detections:
    det = {"xyxy": [x1,y1,x2,y2], "cls": int, "conf": float, "poly": np.ndarray (optional)}
    """
    if not dets:
        return []
    kept = []
    # group by class to avoid suppressing different classes
    by_cls = {}
    for d in dets:
        cid = int(d["cls"])
        by_cls.setdefault(cid, []).append(d)

    for cid, group in by_cls.items():
        group = sorted(group, key=lambda d: d["conf"], reverse=True)
        suppressed = [False] * len(group)
        for i in range(len(group)):
            if suppressed[i]:
                continue
            base = group[i]
            kept.append(base)
            box_i = base["xyxy"]
            for j in range(i + 1, len(group)):
                if suppressed[j]:
                    continue
                if iou_box(box_i, group[j]["xyxy"]) > iou_thres:
                    suppressed[j] = True
    return kept

# ---- Simple IoU-based tracker ----
def update_tracks(tracks, objects, frame_idx, iou_thres=0.3, max_age=15):
    """
    tracks: list of dicts:
        { 'id': int, 'bbox': [x1,y1,x2,y2], 'label': str, 'last_seen': int }
    objects: list of dicts (current frame), each will get 'track_id' attached:
        { 'bbox', 'poly', 'label', 'score', ... }
    frame_idx: current frame index
    """
    # Mark all tracks as unmatched initially
    for tr in tracks:
        tr["matched"] = False

    # For each object, try to match to an existing track
    for obj in objects:
        best_iou = 0.0
        best_track = None
        ob_box = obj["bbox"]
        ob_label = obj["label"]

        for tr in tracks:
            # Prefer same coarse label to keep identities reasonable
            if tr["label"] != ob_label:
                continue
            iou = iou_box(ob_box, tr["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_track = tr

        if best_track is not None and best_iou >= iou_thres:
            # Attach this object to the best track
            obj["track_id"] = best_track["id"]
            best_track["bbox"] = ob_box
            best_track["label"] = ob_label
            best_track["last_seen"] = frame_idx
            best_track["matched"] = True
        else:
            # No suitable track -> create new one later
            obj["track_id"] = None

    # Any object without track_id gets a new track
    next_id = (max([t["id"] for t in tracks], default=0) + 1) if tracks else 1
    for obj in objects:
        if obj["track_id"] is None:
            obj["track_id"] = next_id
            tracks.append({
                "id": next_id,
                "bbox": obj["bbox"],
                "label": obj["label"],
                "last_seen": frame_idx,
                "matched": True,
            })
            next_id += 1

    # Remove stale tracks
    tracks[:] = [
        tr for tr in tracks
        if (frame_idx - tr["last_seen"]) <= max_age
    ]

    return tracks, objects

# ------------------------
# Offline second pass: save original + coarse, no skipping
# ------------------------
def save_original_and_coarse_noskip(args,
                                    yolo,
                                    clf,
                                    coarse_classes,
                                    class_filter):
    source = args.source
    if source.isdigit():
        print("[WARN] Source is a webcam index. Skipping offline no-skip save (no finite video).")
        return

    cap2 = cv2.VideoCapture(source)
    if not cap2.isOpened():
        print(f"[WARN] Could not reopen source '{source}' for offline save.")
        return

    src_fps2 = cap2.get(cv2.CAP_PROP_FPS)
    if src_fps2 is None or src_fps2 <= 0 or np.isnan(src_fps2):
        src_fps2 = 30.0

    out_w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    out_h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    Hc, Wc = args.clf_img_size

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    fourcc_auto = cv2.VideoWriter_fourcc(*"mp4v")

    orig_path = output_dir / "original.mp4"
    coarse_path = output_dir / "overlay_coarse.mp4"

    orig_writer = cv2.VideoWriter(str(orig_path), fourcc_auto, src_fps2, (out_w2, out_h2))
    if not orig_writer.isOpened():
        print(f"[WARN] Could not open offline writer for '{orig_path}'. Skipping original.mp4.")
        orig_writer = None

    coarse_writer = cv2.VideoWriter(str(coarse_path), fourcc_auto, src_fps2, (out_w2, out_h2))
    if not coarse_writer.isOpened():
        print(f"[WARN] Could not open offline writer for '{coarse_path}'. Skipping overlay_coarse.mp4.")
        coarse_writer = None

    tile_rows, tile_cols = args.tiles
    use_tiling = (tile_rows > 1 or tile_cols > 1)
    tile_overlap = max(0.0, min(0.5, float(args.tile_overlap)))

    tracks = []
    frame_idx = 0

    print("[INFO] Offline pass: writing original.mp4 and overlay_coarse.mp4 with NO frame skips...")

    while True:
        ok, frame = cap2.read()
        if not ok:
            break
        frame_idx += 1
        H, W = frame.shape[:2]

        if orig_writer is not None:
            orig_writer.write(frame)

        detections = []

        # --- YOLO detection (with segmentation), NO skipping, similar to main loop ---
        if not use_tiling:
            results = yolo.predict(
                source=frame,
                imgsz=args.yolo_imgsz,
                conf=args.yolo_conf,
                iou=args.yolo_iou,
                classes=class_filter,
                max_det=args.yolo_max_det,
                verbose=False
            )
            res = results[0]
            boxes = res.boxes
            masks_obj = getattr(res, "masks", None)

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy().astype(float)

                polys = None
                if masks_obj is not None:
                    try:
                        polys = masks_obj.xy
                    except Exception:
                        polys = None

                for idx, (bb, cid, conf) in enumerate(zip(xyxy, cls_ids, confs)):
                    x1, y1, x2, y2 = clip_xyxy(bb, W, H)
                    poly = None
                    if polys is not None and idx < len(polys):
                        poly = polys[idx]
                        if poly is not None:
                            poly = poly.astype(np.int32)

                    detections.append({
                        "xyxy": [x1, y1, x2, y2],
                        "cls": int(cid),
                        "conf": float(conf),
                        "poly": poly,
                    })
        else:
            tile_w = max(1, W // tile_cols)
            tile_h = max(1, H // tile_rows)
            overlap_x = int(tile_overlap * tile_w)
            overlap_y = int(tile_overlap * tile_h)

            for tr in range(tile_rows):
                for tc in range(tile_cols):
                    x1_tile = max(0, tc * tile_w - overlap_x)
                    y1_tile = max(0, tr * tile_h - overlap_y)
                    x2_tile = min(W, (tc + 1) * tile_w + overlap_x)
                    y2_tile = min(H, (tr + 1) * tile_h + overlap_y)

                    tile = frame[y1_tile:y2_tile, x1_tile:x2_tile]
                    if tile.size == 0:
                        continue

                    results = yolo.predict(
                        source=tile,
                        imgsz=args.yolo_imgsz,
                        conf=args.yolo_conf,
                        iou=args.yolo_iou,
                        classes=class_filter,
                        max_det=args.yolo_max_det,
                        verbose=False
                    )
                    res = results[0]
                    boxes = res.boxes
                    masks_obj = getattr(res, "masks", None)

                    if boxes is None or len(boxes) == 0:
                        continue

                    xyxy = boxes.xyxy.cpu().numpy()
                    cls_ids = boxes.cls.cpu().numpy().astype(int)
                    confs = boxes.conf.cpu().numpy().astype(float)

                    polys = None
                    if masks_obj is not None:
                        try:
                            polys = masks_obj.xy
                        except Exception:
                            polys = None

                    for idx, (bb, cid, conf) in enumerate(zip(xyxy, cls_ids, confs)):
                        tx1, ty1, tx2, ty2 = bb
                        fx1 = x1_tile + int(tx1)
                        fy1 = y1_tile + int(ty1)
                        fx2 = x1_tile + int(tx2)
                        fy2 = y1_tile + int(ty2)
                        fx1, fy1, fx2, fy2 = clip_xyxy((fx1, fy1, fx2, fy2), W, H)

                        poly = None
                        if polys is not None and idx < len(polys):
                            poly = polys[idx]
                            if poly is not None:
                                poly = poly.astype(np.int32)
                                poly[:, 0] += x1_tile
                                poly[:, 1] += y1_tile

                        detections.append({
                            "xyxy": [fx1, fy1, fx2, fy2],
                            "cls": int(cid),
                            "conf": float(conf),
                            "poly": poly,
                        })

            detections = nms_detections(detections, iou_thres=max(0.6, args.yolo_iou))

        # --- Classification and object building (coarse only) ---
        objects = []
        if detections:
            for det in detections:
                x1, y1, x2, y2 = det["xyxy"]
                poly = det.get("poly", None)

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                x = preprocess_for_classifier(crop, (Hc, Wc))
                pred = clf.predict(x, verbose=0)

                if isinstance(pred, (list, tuple)):
                    coarse_vec = pred[-1][0]
                else:
                    coarse_vec = pred[0]

                coarse_vec = np.asarray(coarse_vec).reshape(-1)

                if coarse_vec.size == 1:
                    p_valid = float(coarse_vec[0])
                    if p_valid >= 0.5:
                        coarse_label = "valid"
                        coarse_score = p_valid
                    else:
                        coarse_label = "invalid_nontarget"
                        coarse_score = 1.0 - p_valid
                else:
                    c_idx = int(np.argmax(coarse_vec))
                    if c_idx < len(coarse_classes):
                        coarse_label = coarse_classes[c_idx]
                    else:
                        coarse_label = f"class_{c_idx}"
                    coarse_score = float(coarse_vec[c_idx])

                if coarse_label == "invalid_background":
                    continue

                objects.append({
                    "bbox": [x1, y1, x2, y2],
                    "poly": poly,
                    "label": coarse_label,
                    "score": coarse_score,
                })

        if objects:
            tracks, objects = update_tracks(
                tracks,
                objects,
                frame_idx,
                iou_thres=float(args.track_iou),
                max_age=int(args.track_max_age),
            )

        # --- Draw overlays for coarse video ---
        if coarse_writer is not None:
            overlay = frame.copy()
            font_scale = float(args.font_scale)
            thickness = int(args.thickness)

            for obj in objects:
                x1, y1, x2, y2 = obj["bbox"]
                poly = obj.get("poly", None)
                coarse_label = obj["label"]
                coarse_score = obj["score"]
                track_id = obj.get("track_id", -1)

                if coarse_label == "valid":
                    color = (40, 200, 40)
                elif coarse_label == "invalid_nontarget":
                    color = (0, 0, 255)
                else:
                    color = (128, 128, 128)

                if poly is not None and poly.size > 0:
                    mask = np.zeros(overlay.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [poly], 1)
                    color_arr = np.array(color, dtype=np.uint8)
                    alpha = 0.5
                    overlay[mask == 1] = (
                        (1 - alpha) * overlay[mask == 1].astype(np.float32) +
                        alpha * color_arr
                    ).astype(np.uint8)
                else:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

                coarse_text = f"#{track_id} {coarse_label}: {coarse_score:.2f}"
                draw_label_box(
                    overlay,
                    coarse_text,
                    (x1, max(0, y1 - 8)),
                    font_scale,
                    thickness,
                    fg=(255, 255, 255),
                    bg=color
                )

            coarse_writer.write(overlay)

    cap2.release()
    if orig_writer is not None:
        orig_writer.release()
    if coarse_writer is not None:
        coarse_writer.release()

    print("[INFO] Offline original.mp4 and overlay_coarse.mp4 saved (no frame skipping).")


# ------------------------
# Main
# ------------------------
def main():
    args = parse_args()

    # Load YOLO detector
    yolo = YOLO(args.yolo_weights)

    # Build name -> id mapping for optional class filtering
    yolo_names = yolo.model.names  # dict id->name
    name_to_id = {name: i for i, name in yolo_names.items()}
    class_filter = None
    if args.yolo_classes:
        wanted = [n.strip() for n in args.yolo_classes.split(",") if n.strip()]
        missing = [n for n in wanted if n not in name_to_id]
        if missing:
            print(f"[WARN] Some YOLO class names not found and will be ignored: {missing}")
        class_filter = [name_to_id[n] for n in wanted if n in name_to_id]
        print(f"[INFO] YOLO restricting to classes: {wanted} -> ids {class_filter}")

    # Load classifier (your Keras model) — we ONLY use the COARSE head in real-time path
    clf = tf.keras.models.load_model(args.model)

    # Load classes from label_map.json
    coarse_classes = load_coarse_classes(args.label_map)
    print(f"[INFO] Coarse classes (from label_map): {coarse_classes}")

    Hc, Wc = args.clf_img_size

    # Open source
    source = args.source
    is_webcam = source.isdigit()
    cap = cv2.VideoCapture(int(source)) if is_webcam else cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open source: {source}", file=sys.stderr)
        sys.exit(1)

    # Determine "real-time" FPS from source (for video files)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps is None or src_fps <= 0 or np.isnan(src_fps):
        src_fps = 30.0
    target_dt = 1.0 / src_fps
    print(f"[INFO] Source FPS (approx): {src_fps:.2f} (target frame time ~ {target_dt*1000:.1f} ms)")
    if args.no_skip:
        print("[INFO] --no-skip enabled: processing EVERY frame (may be slower than real time).")
    else:
        print(f"[INFO] Frame skipping enabled for real-time playback on video files "
              f"(max-skip = {args.max_skip} frames per step).")

    # Common video properties for writers
    out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fps_for_writer = src_fps if not is_webcam else (cap.get(cv2.CAP_PROP_FPS) or 30.0)

    # Optional video writer (original --save argument, real-time coarse overlay)
    writer = None
    if args.save:
        out_path = Path(args.save)
        fourcc = cv2.VideoWriter_fourcc(*("mp4v" if out_path.suffix.lower() == ".mp4" else "XVID"))
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_for_writer, (out_w, out_h))
        if not writer.isOpened():
            print(f"[WARN] Could not open writer for '{out_path}'. Disabling save.")
            writer = None

    last_t = time.time()
    font_scale = float(args.font_scale)
    thickness = int(args.thickness)

    tile_rows, tile_cols = args.tiles
    use_tiling = (tile_rows > 1 or tile_cols > 1)
    tile_overlap = max(0.0, min(0.5, float(args.tile_overlap)))

    # Tracking state
    tracks = []
    frame_idx = 0

    print("[INFO] Press 'q' to quit window.")
    if use_tiling:
        print(f"[INFO] Tiled detection enabled: {tile_rows}x{tile_cols}, overlap={tile_overlap:.2f}")

    # -------- REAL-TIME LOOP (coarse overlay + preview) --------
    while True:
        frame_start = time.time()
        frame_idx += 1

        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]

        detections = []

        # --- YOLO detection (with segmentation) ---
        if not use_tiling:
            # Single-shot on full frame
            results = yolo.predict(
                source=frame,
                imgsz=args.yolo_imgsz,
                conf=args.yolo_conf,
                iou=args.yolo_iou,
                classes=class_filter,
                max_det=args.yolo_max_det,
                verbose=False
            )
            res = results[0]
            boxes = res.boxes
            masks_obj = getattr(res, "masks", None)

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy().astype(float)

                # If segmentation masks are available, use polygons
                polys = None
                if masks_obj is not None:
                    try:
                        polys = masks_obj.xy  # list of (N_i, 2) arrays in image coords
                    except Exception:
                        polys = None

                for idx, (bb, cid, conf) in enumerate(zip(xyxy, cls_ids, confs)):
                    x1, y1, x2, y2 = clip_xyxy(bb, W, H)
                    poly = None
                    if polys is not None and idx < len(polys):
                        poly = polys[idx]
                        if poly is not None:
                            poly = poly.astype(np.int32)

                    detections.append({
                        "xyxy": [x1, y1, x2, y2],
                        "cls": int(cid),
                        "conf": float(conf),
                        "poly": poly,
                    })
        else:
            # Tiled detection for small/packed parts
            tile_w = max(1, W // tile_cols)
            tile_h = max(1, H // tile_rows)
            overlap_x = int(tile_overlap * tile_w)
            overlap_y = int(tile_overlap * tile_h)

            for tr in range(tile_rows):
                for tc in range(tile_cols):
                    x1_tile = max(0, tc * tile_w - overlap_x)
                    y1_tile = max(0, tr * tile_h - overlap_y)
                    x2_tile = min(W, (tc + 1) * tile_w + overlap_x)
                    y2_tile = min(H, (tr + 1) * tile_h + overlap_y)

                    tile = frame[y1_tile:y2_tile, x1_tile:x2_tile]
                    if tile.size == 0:
                        continue

                    results = yolo.predict(
                        source=tile,
                        imgsz=args.yolo_imgsz,
                        conf=args.yolo_conf,
                        iou=args.yolo_iou,
                        classes=class_filter,
                        max_det=args.yolo_max_det,
                        verbose=False
                    )
                    res = results[0]
                    boxes = res.boxes
                    masks_obj = getattr(res, "masks", None)

                    if boxes is None or len(boxes) == 0:
                        continue

                    xyxy = boxes.xyxy.cpu().numpy()
                    cls_ids = boxes.cls.cpu().numpy().astype(int)
                    confs = boxes.conf.cpu().numpy().astype(float)

                    polys = None
                    if masks_obj is not None:
                        try:
                            polys = masks_obj.xy  # list of polygons in tile coords
                        except Exception:
                            polys = None

                    for idx, (bb, cid, conf) in enumerate(zip(xyxy, cls_ids, confs)):
                        # Map tile coords back to full-frame coords
                        tx1, ty1, tx2, ty2 = bb
                        fx1 = x1_tile + int(tx1)
                        fy1 = y1_tile + int(ty1)
                        fx2 = x1_tile + int(tx2)
                        fy2 = y1_tile + int(ty2)
                        fx1, fy1, fx2, fy2 = clip_xyxy((fx1, fy1, fx2, fy2), W, H)

                        poly = None
                        if polys is not None and idx < len(polys):
                            poly = polys[idx]
                            if poly is not None:
                                poly = poly.astype(np.int32)
                                # Shift polygon from tile space to full-frame space
                                poly[:, 0] += x1_tile
                                poly[:, 1] += y1_tile

                        detections.append({
                            "xyxy": [fx1, fy1, fx2, fy2],
                            "cls": int(cid),
                            "conf": float(conf),
                            "poly": poly,
                        })

            # Merge overlapping detections across tiles (NMS with slightly high IoU)
            detections = nms_detections(detections, iou_thres=max(0.6, args.yolo_iou))

        # FPS (processing FPS, not necessarily real-time FPS)
        now = time.time()
        fps = 1.0 / max(1e-6, now - last_t)
        last_t = now

        # --- CLASSIFY (COARSE ONLY) + BUILD OBJECTS ---
        objects = []  # objects that survive (no invalid_background)
        if detections:
            for det in detections:
                x1, y1, x2, y2 = det["xyxy"]
                poly = det.get("poly", None)

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                x = preprocess_for_classifier(crop, (Hc, Wc))
                pred = clf.predict(x, verbose=0)

                # Only coarse info is used in real-time path
                if isinstance(pred, (list, tuple)):
                    coarse_vec = pred[-1][0]
                else:
                    coarse_vec = pred[0]

                coarse_vec = np.asarray(coarse_vec).reshape(-1)

                # Interpret coarse head
                if coarse_vec.size == 1:
                    # Binary fallback: >0.5 -> valid, else invalid_nontarget
                    p_valid = float(coarse_vec[0])
                    if p_valid >= 0.5:
                        coarse_label = "valid"
                        coarse_score = p_valid
                    else:
                        coarse_label = "invalid_nontarget"
                        coarse_score = 1.0 - p_valid
                else:
                    # Multi-class softmax, assume order matches coarse_classes from label_map
                    c_idx = int(np.argmax(coarse_vec))
                    if c_idx < len(coarse_classes):
                        coarse_label = coarse_classes[c_idx]
                    else:
                        coarse_label = f"class_{c_idx}"
                    coarse_score = float(coarse_vec[c_idx])

                # Skip "invalid_background" entirely
                if coarse_label == "invalid_background":
                    continue

                objects.append({
                    "bbox": [x1, y1, x2, y2],
                    "poly": poly,
                    "label": coarse_label,
                    "score": coarse_score,
                })

        # --- UPDATE TRACKS ---
        if objects:
            tracks, objects = update_tracks(
                tracks,
                objects,
                frame_idx,
                iou_thres=float(args.track_iou),
                max_age=int(args.track_max_age),
            )

        # --- DRAW (mask overlay) ---
        for obj in objects:
            x1, y1, x2, y2 = obj["bbox"]
            poly = obj.get("poly", None)
            coarse_label = obj["label"]
            coarse_score = obj["score"]
            track_id = obj.get("track_id", -1)

            # Color based on coarse label
            if coarse_label == "valid":
                color = (40, 200, 40)          # green-ish
            elif coarse_label == "invalid_nontarget":
                color = (0, 0, 255)            # RED as requested
            else:
                color = (128, 128, 128)        # fallback

            # If we have a polygon mask, overlay color on object shape
            if poly is not None and poly.size > 0:
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 1)
                color_arr = np.array(color, dtype=np.uint8)

                # Alpha-blend color onto the masked region (semi-transparent)
                alpha = 0.5
                frame[mask == 1] = (
                    (1 - alpha) * frame[mask == 1].astype(np.float32) +
                    alpha * color_arr
                ).astype(np.uint8)
            else:
                # Fallback: filled rectangle if no mask is available
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)

            # Single-line label: [ID] coarse label + score
            line = f"#{track_id} {coarse_label}: {coarse_score:.2f}"
            draw_label_box(frame, line, (x1, max(0, y1 - 8)),
                           font_scale, thickness,
                           fg=(255,255,255), bg=color)

        # HUD: processing FPS + source FPS
        draw_label_box(frame,
                       f"Proc FPS: {fps:.1f} | Src FPS: {src_fps:.1f}",
                       (10, 30),
                       font_scale, thickness, fg=(255,255,255), bg=(0,0,0))

        # --- WRITE COARSE OVERLAY VIDEO (real-time path, only if user explicitly requested via --save) ---
        if writer is not None:
            writer.write(frame)

        # Display LAST, so saving doesn't affect real-time responsiveness as much
        if not args.no_display:
            cv2.imshow("YOLO-seg + Coarse Classifier + Tracking (press 'q' to quit)", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break

        # --- REAL-TIME FRAME SKIP LOGIC (video files only, unless --no-skip) ---
        proc_dt = time.time() - frame_start
        if (not is_webcam) and (not args.no_skip) and (args.max_skip != 0):
            if proc_dt > target_dt:
                # How many frame-intervals did this one iteration consume?
                skips = int(proc_dt / target_dt) - 1
                if skips > 0:
                    if args.max_skip > 0:
                        skips = min(skips, args.max_skip)
                    # Grab+discard a few frames to catch up with "real time"
                    for _ in range(skips):
                        cap.grab()

    # -------- CLEANUP REAL-TIME RESOURCES --------
    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    # -------- OFFLINE SECOND PASS: NO-SKIP ORIGINAL + COARSE OVERLAYS --------
    if not args.no_autosave_output and (not is_webcam):
        save_original_and_coarse_noskip(
            args=args,
            yolo=yolo,
            clf=clf,
            coarse_classes=coarse_classes,
            class_filter=class_filter,
        )

    print("[INFO] Done.")

if __name__ == "__main__":
    main()
