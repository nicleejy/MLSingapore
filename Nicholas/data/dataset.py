import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


ROOT_DIR = os.path.dirname(__file__)

annotations_input_dir = os.path.join(
    ROOT_DIR, "../../data/external/FoodSeg103/Images/ann_dir"
)
images_input_dir = os.path.join(
    ROOT_DIR, "../../data/external/FoodSeg103/Images/img_dir"
)
annotations_output_dir = os.path.join(
    ROOT_DIR, "../../data/processed/Yolo_FoodSeg103/labels"
)


def draw_normalised_polygons(image_path, polygons, class_id):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    overlay_img = img.copy()
    denormalised_polygon = [(int(x * width), int(y * height)) for x, y in polygons]
    pts = np.array(denormalised_polygon, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(overlay_img, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
    overlay_img_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 6))
    plt.imshow(overlay_img_rgb)
    plt.title("class ID " + str(class_id))
    plt.axis("off")
    plt.show()


# To convert each annotation mask into a YOLO-compatible format
def mask_to_polygons(mask_path, output_path):
    img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    height, width = img.shape[:2]
    unique_classes = np.unique(img)
    file = open(output_path, "w+")
    for class_index in unique_classes:
        if class_index == 0:
            continue
        mask = np.uint8(img == class_index)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            normalised = [
                (point[0][0] / width, point[0][1] / height) for point in contour
            ]
            flattened = [coord for pair in normalised for coord in pair]
            if flattened:
                file.write(f"{class_index} " + " ".join(map(str, flattened)) + "\n")


# Convert a folder of masks to YOLO-compatable format
def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(input_dir, filename)
            output_path = os.path.join(
                output_dir, os.path.splitext(filename)[0] + ".txt"
            )
            mask_to_polygons(image_path, output_path)
            print(f"Processed {filename}")


# train_test_sets = ["train", "test"]
# for set_name in train_test_sets:
#     input_directory = os.path.join(annotations_input_dir, set_name)
#     output_directory = os.path.join(annotations_output_dir, set_name)
#     process_directory(input_directory, output_directory)
