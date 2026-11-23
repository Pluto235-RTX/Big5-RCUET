from ultralytics import YOLO 
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
model = YOLO(BASE_DIR/"runs"/"my_project"/"obj_detecter"/"weights"/"best.pt")
metrics = model.val(
    data = BASE_DIR/"DATASET"/"data.yaml",
    split = "test",
    imgsz = 640,
    batch = 8, 
    plots = True,
    project = BASE_DIR/"runs"/"evaluation",
    name = "results"
)
print("Precision: ", metrics.box.mp)
print("Recall: ", metrics.box.mr)
print("mAP50: ", metrics.box.map50) 
print("mAP75: ", metrics.box.map75)
print("mAP50-95: ", metrics.box.map)

p = metrics.box.mp
r = metrics.box.mr
f1 = 2 * p * r / (p + r) if p + r > 0 else 0
print("f1: ",f1)