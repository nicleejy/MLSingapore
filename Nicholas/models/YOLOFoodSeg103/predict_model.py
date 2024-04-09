from ultralytics import YOLO
import os


ROOT_DIR = os.path.dirname(__file__)

model_path = os.path.join(ROOT_DIR, "results/v1-4/weights/last.pt")

model = YOLO(model_path)
inference_folder = os.path.join(ROOT_DIR, "infer")
inference_folder = r"E:\MLSingapore\MLSingapore\data\external\foodsg-233\Rice, chicken katsu with Japanese curry"
image_paths = [os.path.join(inference_folder, f) for f in os.listdir(inference_folder)]


results = model(image_paths[0:20])

for result in results:
    boxes = result.boxes
    masks = result.masks
    probs = result.probs
    result.show()
