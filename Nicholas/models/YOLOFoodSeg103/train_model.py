import torch
import os
from ultralytics import YOLO


ROOT_DIR = os.path.dirname(__file__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = YOLO("yolov8n-seg.pt")
config_path = os.path.join(ROOT_DIR, "config.yaml")
project_path = os.path.join(ROOT_DIR, "results")
name = "v1-"
model = model.to(device)


if __name__ == "__main__":
    n_epochs = 20
    bs = -1
    results = model.train(
        data=config_path,
        epochs=n_epochs,
        imgsz=640,
        project=project_path,
        name=name,
        patience=10,
    )
