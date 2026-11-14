"""Fine-tune YOLO11 on the vehicle dataset (Roboflow export in YOLO format).

Usage:
    python train.py --data data.yaml --model yolo11n.pt --imgsz 1024 --epochs 100
"""
import argparse

from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--data", required=True, help="dataset data.yaml")
p.add_argument("--model", default="yolo11n.pt")
p.add_argument("--imgsz", type=int, default=640)
p.add_argument("--epochs", type=int, default=100)
args = p.parse_args()

model = YOLO(args.model)
model.train(data=args.data, imgsz=args.imgsz, epochs=args.epochs)
print(model.val().box.map50)  # mAP50 on the validation split
