from pathlib import Path
from ultralytics import YOLO
model = YOLO("yolo26n.pt")
BASE_DIR = Path(__file__).resolve().parent
results = model.train(
    data = BASE_DIR/"DATASET"/"data.yaml",
    epochs = 5,
    imgsz = 640,
    batch = 8, 
    project = BASE_DIR/"runs"/"my_project",
    name = "obj_detecter"
)