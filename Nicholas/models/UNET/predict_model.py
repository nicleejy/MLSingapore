import cv2
import numpy as np
import colorsys
import numpy as np
import json
import os
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from model import UNET
from utils import load_checkpoint

ROOT_DIR = os.path.dirname(__file__)

json_file_path = os.path.join(
    ROOT_DIR, "../../../data/interim/foodseg103_classmap.json"
)
annotations_input_dir = os.path.join(
    ROOT_DIR, "../../../data/external/FoodSeg103/Images/ann_dir"
)

DEVICE = "cuda"
IMAGE_MAX_SIZE = IMAGE_HEIGHT = IMAGE_WIDTH = 400


def get_class_color_mappings(n):
    results = {}
    for i in range(n):
        hue = i / n
        saturation = 0.9
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        results[i] = tuple(int(x * 255) for x in rgb)
    return results


def create_colored_overlay_with_text(mask, color_mappings, class_mappings):
    colored_overlay = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_index, color in color_mappings.items():
        if class_index == 0:
            continue
        colored_overlay[mask == class_index] = color
        class_mask = np.zeros(mask.shape, dtype=np.uint8)
        class_mask[mask == class_index] = 1
        contours, _ = cv2.findContours(
            class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(colored_overlay, contours, -1, color, thickness=cv2.FILLED)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                # compute centroid
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(
                    colored_overlay,
                    class_mappings[str(class_index)],
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
    return colored_overlay


def overlay_mask(image_path, mask_path, color_mappings, class_mappings, alpha=0.5):
    mask = np.array(
        Image.open(mask_path),
        dtype=np.float32,
    )
    colored_overlay = create_colored_overlay_with_text(
        mask, color_mappings=color_mappings, class_mappings=class_mappings
    )
    image = cv2.imread(image_path)
    blended_image = cv2.addWeighted(image, 1 - alpha, colored_overlay, alpha, 0)
    cv2.imshow("Result", blended_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def predict(model, image_path, model_path, color_mappings, class_mappings, alpha=0.5):
    image = cv2.imread(image_path)
    inference_transform = A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_MAX_SIZE),
            A.PadIfNeeded(
                min_height=IMAGE_HEIGHT,
                min_width=IMAGE_WIDTH,
                border_mode=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            ),
            A.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )
    image_transform = A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_MAX_SIZE),
            A.PadIfNeeded(
                min_height=IMAGE_HEIGHT,
                min_width=IMAGE_WIDTH,
                border_mode=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            ),
        ]
    )
    transformed_image = inference_transform(image=image)
    transformed_input = transformed_image["image"].unsqueeze(0)
    load_checkpoint(filename=model_path, model=model)
    model.eval()
    with torch.no_grad():
        transformed_input = transformed_input.to(device="cuda")
        preds = model(transformed_input)
        preds = torch.softmax(preds, dim=1)  # [N, C, H, W]
        preds = torch.argmax(preds, dim=1)  # [N, H, W]
        preds = preds.cpu().numpy()[0]

    colored_overlay = create_colored_overlay_with_text(
        preds, color_mappings=color_mappings, class_mappings=class_mappings
    )

    image = image_transform(image=image)["image"]
    blended_image = cv2.addWeighted(image, 1 - alpha, colored_overlay, alpha, 0)
    cv2.imshow("Result", blended_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


num_classes = 104
color_mappings = get_class_color_mappings(num_classes)
with open(json_file_path, "r") as j:
    class_mappings = json.load(j)

model = UNET(in_channels=3, out_channels=104).to(DEVICE)


# overlay_mask(
#     image_path=r"E:\MLSingapore\MLSingapore\data\external\FoodSeg103\Images\img_dir\test\00007034.jpg",
#     mask_path=r"E:\MLSingapore\MLSingapore\data\external\FoodSeg103\Images\ann_dir\test\00007034.png",
#     color_mappings=color_mappings,
#     class_mappings=class_mappings,
#     alpha=0,
# )


inference_folder = (
    r"E:\MLSingapore\MLSingapore\data\external\FoodSeg103\Images\img_dir\test"
)
image_paths = [os.path.join(inference_folder, f) for f in os.listdir(inference_folder)]

for image_path in image_paths:
    predict(
        model=model,
        image_path=image_path,
        model_path=r"E:\MLSingapore\MLSingapore\epoch20.pth.tar",
        color_mappings=color_mappings,
        class_mappings=class_mappings,
    )
