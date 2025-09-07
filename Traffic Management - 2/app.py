#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO + Centroid Tracking vehicle counter (no Flask, no external tracker).

Features
- Unique ID per vehicle using a simple centroid tracker
- Count once when a track crosses a virtual line
- Yellow boxes, ID labels, live HUD, FPS
- Traffic flow (Low/Medium/High) derived from recent counts/min
- Optional save to video
- Playback controls: q=quit, p=pause, -=slower, = (equals)=faster

Install:
    pip install ultralytics==8.2.103 opencv-python numpy

Run:
    python vehicle_counter_yolo.py --source input_video.mp4
    python vehicle_counter_yolo.py --source input_video.mp4 --model yolov8n.pt --save result.mp4
"""

import argparse
import collections
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ---------------------------- CONFIG ---------------------------------

VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "bicycle", "train"}

YELLOW = (0, 255, 255)
WHITE  = (255, 255, 255)
RED    = (0, 0, 255)
GREEN  = (0, 200, 0)
ORANGE = (0, 165, 255)
HUD_BG = (24, 24, 24)


def compute_flow_level(vehicles_per_min: float) -> str:
    """Tune thresholds for your setting."""
    if vehicles_per_min > 15:    # >3 vehicles/sec
        return "High"
    elif vehicles_per_min > 10:   # 1–3 vehicles/sec
        return "Medium"
    else:
        return "Low"


# ------------------------- SIMPLE TRACKER -----------------------------

class CentroidTracker:
    """
    Minimal centroid tracker:
    - Associates detections to existing tracks using nearest centroid (IoU-free).
    - Deregisters tracks that disappear for > max_disappeared frames.
    """
    def __init__(self, max_disappeared=30, max_distance=60):
        self.next_id = 1
        self.objects = {}          # id -> (cx, cy)
        self.bboxes = {}           # id -> (x1,y1,x2,y2)
        self.disappeared = {}      # id -> frames
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox):
        self.objects[self.next_id] = centroid
        self.bboxes[self.next_id] = bbox
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, object_id):
        self.objects.pop(object_id, None)
        self.bboxes.pop(object_id, None)
        self.disappeared.pop(object_id, None)

    def update(self, detections):
        """
        detections: list of (x1,y1,x2,y2) in current frame
        Returns dict: id -> (centroid, bbox)
        """
        if len(detections) == 0:
            # mark everything disappeared
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return {oid: (self.objects[oid], self.bboxes[oid]) for oid in self.objects}

        # compute centroids for new detections
        input_centroids = []
        for (x1, y1, x2, y2) in detections:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            input_centroids.append((cx, cy))

        if len(self.objects) == 0:
            # no existing tracks, register all
            for c, b in zip(input_centroids, detections):
                self.register(c, b)
        else:
            # match old -> new by nearest centroid
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # distance matrix
            D = np.linalg.norm(
                np.array(object_centroids)[:, None, :] - np.array(input_centroids)[None, :, :],
                axis=2
            )

            # greedy match (smallest distances first)
            used_rows = set()
            used_cols = set()
            for _ in range(min(D.shape[0], D.shape[1])):
                r, c = np.unravel_index(np.argmin(D), D.shape)
                if r in used_rows or c in used_cols:
                    D[r, c] = np.inf
                    continue
                if D[r, c] <= self.max_distance:
                    oid = object_ids[r]
                    self.objects[oid] = input_centroids[c]
                    self.bboxes[oid] = detections[c]
                    self.disappeared[oid] = 0
                    used_rows.add(r)
                    used_cols.add(c)
                D[r, c] = np.inf

            # rows (existing) not matched -> disappeared++
            for r, oid in enumerate(object_ids):
                if r not in used_rows:
                    self.disappeared[oid] += 1
                    if self.disappeared[oid] > self.max_disappeared:
                        self.deregister(oid)

            # cols (new detections) not matched -> new IDs
            for c, (cent, box) in enumerate(zip(input_centroids, detections)):
                if c not in used_cols:
                    self.register(cent, box)

        return {oid: (self.objects[oid], self.bboxes[oid]) for oid in self.objects}


# ----------------------------- HUD -----------------------------------

def draw_hud(frame, curr_counts, total_count, flow_text, fps_disp):
    h, w = frame.shape[:2]
    pad = 12
    hud_h = 88
    cv2.rectangle(frame, (0, 0), (w, hud_h), HUD_BG, thickness=-1)

    cv2.putText(frame, "Traffic Monitor (YOLOv8 + CentroidTracker)",
                (pad, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2, cv2.LINE_AA)

    flow_color = GREEN if flow_text == "Low" else ORANGE if flow_text == "Medium" else RED
    cv2.putText(frame, f"Flow: {flow_text}",
                (pad, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, flow_color, 2, cv2.LINE_AA)

    cv2.putText(frame, f"FPS: {fps_disp:.1f}",
                (w - 140, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)

    cv2.putText(frame, f"Counted: {total_count}",
                (w - 180, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)

    # Current per-frame visible counts by class (small summary)
    if curr_counts:
        curr_str = ", ".join([f"{k}:{curr_counts[k]}" for k in sorted(curr_counts)])
    else:
        curr_str = "(none)"
    cv2.putText(frame, f"Now: {curr_str}", (pad, hud_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)


# ----------------------------- MAIN ----------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, required=True, help="Path to video file")
    ap.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model (yolov8n.pt / yolov5s.pt / custom.pt)")
    ap.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference size")
    ap.add_argument("--save", type=str, default="", help="Optional output video path")
    ap.add_argument("--line", type=float, default=0.60, help="Horizontal count line as fraction of height (0-1)")
    ap.add_argument("--display_width", type=int, default=960, help="Resize display window to this width (keeps aspect)")
    args = ap.parse_args()

    src_path = Path(args.source)
    if not src_path.exists():
        raise FileNotFoundError(f"Video not found: {src_path}")

    print(f"[i] Loading model: {args.model}")
    model = YOLO(args.model)

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {src_path}")

    base_fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_y    = int(args.line * height)

    # Optional writer
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, base_fps, (width, height))
        print(f"[i] Saving processed video to: {args.save}")

    tracker = CentroidTracker(max_disappeared=5, max_distance=60)
    track_crossed = set()         # IDs already counted
    total_count = 0

    # Moving window of counts to estimate flow (timestamps of counted events)
    recent_events = collections.deque(maxlen=600)  # store timestamps of last N counted crossings

    prev_t = time.time()
    fps_disp = 0.0
    paused = False
    speed_scale = 1.0  # >1 = slower playback; <1 = faster

    print("[i] Controls: q=quit, p=pause, -=slower, = (equals)=faster")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[i] End of video.")
                break
        else:
            ret = True

        # YOLO inference
        results = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)

        detections = []
        curr_counts = collections.Counter()

        if results and len(results) > 0 and results[0].boxes is not None:
            names = model.names  # id->name
            for b in results[0].boxes:
                cls_id = int(b.cls)
                cls_name = names[cls_id] if isinstance(names, (list, dict)) else str(cls_id)
                if cls_name not in VEHICLE_CLASSES:
                    continue
                conf = float(b.conf)
                if conf <0.5:
                    continue
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                detections.append((x1, y1, x2, y2))
                curr_counts[cls_name] += 1

        # Update tracker with current detections
        tracks = tracker.update(detections)  # id -> (centroid, bbox)

        # Draw count line
        cv2.line(frame, (0, line_y), (width, line_y), (255, 0, 0), 2)

        # For each active track, draw and check crossing
        for oid, (centroid, bbox) in tracks.items():
            x1, y1, x2, y2 = bbox
            cx, cy = centroid

            # draw yellow box + ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), YELLOW, 3)
            cv2.putText(frame, f"ID:{oid}", (x1, max(22, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 4, YELLOW, -1)

            # crossing logic: if the bbox center crosses the line and wasn't counted before
            # we check previous centroid (stored in tracker's objects history implicitly by update order)
            # Simple heuristic: if |cy - line_y| < threshold and not counted -> count it
            if oid not in track_crossed:
                if (cy >= line_y - 5) and (cy <= line_y + 5):
                    total_count += 1
                    track_crossed.add(oid)
                    recent_events.append(time.time())

        # Flow estimate
        now = time.time()
        # remove old events (>60 sec)
        while recent_events and (now - recent_events[0] > 60.0):
            recent_events.popleft()
        vehicles_per_min = float(len(recent_events))  # events in last 60s
        flow_text = compute_flow_level(vehicles_per_min)

        # FPS smoothing
        dt = now - prev_t
        prev_t = now
        if dt > 0:
            fps_disp = 0.9 * fps_disp + 0.1 * (1.0 / dt) if fps_disp > 0 else (1.0 / dt)

        # HUD
        draw_hud(frame, curr_counts, total_count, flow_text, fps_disp)

        # Save exactly what we show (original size)
        if writer is not None:
            writer.write(frame)

        # Resize for display so any video fits on screen nicely
        if args.display_width > 0 and width > 0:
            scale = args.display_width / float(width)
            disp_h = int(height * scale)
            disp = cv2.resize(frame, (args.display_width, disp_h), interpolation=cv2.INTER_AREA)
        else:
            disp = frame

        cv2.imshow("YOLO Vehicle Counter", disp)

        # Playback pacing
        delay_ms = int(max(1, (1000.0 / base_fps) * speed_scale))
        key = cv2.waitKey(delay_ms) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('-'):
            speed_scale = min(4.0, speed_scale + 0.25)  # slower
        elif key == ord('='):
            speed_scale = max(0.25, speed_scale - 0.25)  # faster

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[i] Saved processed video to: {args.save}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
