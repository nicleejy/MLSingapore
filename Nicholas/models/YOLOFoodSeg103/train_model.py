import torch
import os
from ultralytics import YOLO


ROOT_DIR = os.path.dirname(__file__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = os.path.join(ROOT_DIR, "results/v1-4/weights/last.pt")

# model = YOLO("yolov8s-seg.pt")
model = YOLO(model=model_path)


config_path = os.path.join(ROOT_DIR, "config.yaml")
project_path = os.path.join(ROOT_DIR, "results")
name = "v1-"
model = model.to(device)


if __name__ == "__main__":
    # n_epochs = 100
    # bs = -1
    # results = model.train(
    #     data=config_path,
    #     epochs=n_epochs,
    #     imgsz=640,
    #     project=project_path,
    #     name=name,
    #     patience=10,
    #     lr0=0.05,
    # )

    model.tune(
        data=config_path,
        epochs=20,
        iterations=100,
        optimizer="AdamW",
        plots=False,
        save=False,
        val=False,
        lr0=0.05,
        patience=10,
    )
