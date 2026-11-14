"""Vehicle counting: YOLO11 detection + DeepSORT tracking + line crossing.

Usage:
    python count_vehicles.py --weights best.pt --source video.mp4 --out out.mp4

The source can be a video file or an HLS/RTSP stream URL.
"""
import argparse

import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--weights", required=True, help="trained YOLO11 weights (best.pt)")
p.add_argument("--source", required=True, help="video file or stream URL")
p.add_argument("--out", default="out.mp4")
p.add_argument("--conf", type=float, default=0.3)
p.add_argument("--line-y", type=float, default=0.62, help="counting line height as a fraction of the frame")
p.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
p.add_argument("--ignore", nargs="*", default=["people"], help="class names not to count")
args = p.parse_args()

model = YOLO(args.weights)
tracker = DeepSort(max_age=30)  # keep a lost track for 30 frames before dropping it
names = model.names

cap = cv2.VideoCapture(args.source)
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

A, B = (0, int(H * args.line_y)), (W, int(H * args.line_y))  # counting line A -> B


def side(p):
    """Sign of the cross product (B - A) x (P - A): which side of the line P is on."""
    return (B[0] - A[0]) * (p[1] - A[1]) - (B[1] - A[1]) * (p[0] - A[0])


last_side, counted = {}, set()
counts = {"in": 0, "out": 0}

n = 0
while not args.max_frames or n < args.max_frames:
    ok, frame = cap.read()
    if not ok:
        break

    res = model(frame, conf=args.conf, verbose=False)[0]
    dets = [([x1, y1, x2 - x1, y2 - y1], c, names[int(k)])  # DeepSORT expects [left, top, w, h]
            for (x1, y1, x2, y2), c, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist())
            if names[int(k)] not in args.ignore]

    for t in tracker.update_tracks(dets, frame=frame):
        if not t.is_confirmed() or t.time_since_update > 0:
            continue
        x1, y1, x2, y2 = map(int, t.to_ltrb())
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        s = side(center)
        prev = last_side.get(t.track_id)
        if prev is not None and prev * s < 0 and t.track_id not in counted:  # sign flipped = crossed the line
            counts["in" if s > 0 else "out"] += 1  # moving down = towards the camera
            counted.add(t.track_id)
        last_side[t.track_id] = s

        color = (0, 200, 0) if t.track_id in counted else (255, 160, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"#{t.track_id} {t.get_det_class()}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.line(frame, A, B, (0, 0, 255), 2)
    cv2.putText(frame, f"IN: {counts['in']}  OUT: {counts['out']}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    out.write(frame)
    n += 1

cap.release()
out.release()
print("frames:", n, "counts:", counts)
