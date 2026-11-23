from ultralytics import YOLO 
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
model = YOLO(BASE_DIR/"runs"/"my_project"/"obj_detecter"/"weights"/"best.pt")
results = model.predict(
    source = BASE_DIR/"SOURCE"/"pic.jpg",
    imgsz = 640,
    conf = 0.25,
    project = BASE_DIR/"runs",
    name = "results",
    save = True
) 
for result in results:
    result.show()