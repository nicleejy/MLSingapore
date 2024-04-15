import torch
import os
import gdown
import zipfile
from ultralytics import YOLO

ROOT_DIR = os.path.dirname(__file__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = YOLO("yolov8n-cls.pt")
config_path = os.path.join(ROOT_DIR, "config.yaml")
project_path = os.path.join(ROOT_DIR, "results")
name = "v1-"
model = model.to(device)

if __name__ == "__main__":
    
    # gdown.download(id='1na7VT3ywXAOqw9_VL19dyQIyS_8EObTF', output='dataset.zip') # small dataset
    # gdown.download(id='1ttuQajCUoAZ0rFLqi6KVad-kmjShtE_9', output='dataset.zip') # full dataset
    with zipfile.ZipFile('dataset.zip', 'r') as zip_ref:
        zip_ref.extractall('datasets')

    n_epochs = 48
    bs = 48
    results = model.train(
        data="datasets",
        epochs=n_epochs,
        imgsz=640,
        project=project_path,
        name=name,
        patience=10,
        batch=bs,
        # cache=True,
        verbose=True
    )