from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

model.train(
    data="/Users/justicealuu/research_project/perovskite_segment/data/solarpark/dataset.yaml",
    epochs=30,
    imgsz=320,
    batch=16,
    device="mps"
)

model.val()