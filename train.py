from ultralytics import YOLO

model = YOLO("models/yolo12n.pt")


results = model.train(
    data="dataset_for_train/data.yaml",
    epochs=250,
    imgsz=640,
    batch=16,
    augment=True,
    name="yolo12_custom"
)