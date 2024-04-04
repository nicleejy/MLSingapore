import pandas as pd
import numpy as np
import os
import cv2

# Get the current directory of this file
CURR_DIR = os.path.dirname(__file__)

# Specify the path to the image data directory [Change to your own path]
DATA_DIR = "../../data/imagery/realsense_overhead"

# [HARDCODED VALUE] - current image size, without augmentation 
IMAGE_SIZE = (480, 640, 3)

'''
Given an image_path, return the RGB values of the image
'''
def process_rgb_image(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_image

'''
Retrieve all RGB images from DATA_DIR
Convert images to its RGB values
Returns a dictionary, where
    key is dish_id
    value is RBG values
'''
def get_dishes_rgb_image():
    processed_images = {}
    for directory in os.listdir(DATA_DIR):
        directory_path = os.path.join(DATA_DIR, directory)

        if os.path.isdir(directory_path):
            rgb_image_path = os.path.join(directory_path, "rgb.png")
            if os.path.isfile(rgb_image_path):
                rgb_image = process_rgb_image(rgb_image_path)
                processed_images[directory] = rgb_image
            else:
                print(f"Warning: RGB image not found in directory {directory}")
    return processed_images

'''
Given an image dictionary, where
    key is dish_id
    value is RGB values
Return DataFrame with 
    first column - dish_id
    second column - flatten array of RGB values
'''
def convert_dishes_rgb_image_to_dataframe(images):
    return pd.DataFrame([(k, v.flatten()) for k, v in images.items()], columns=['dish_id', 'image'])

'''
Given a flatten image,
Return original shape of the image.
'''
def convert_flatten_image_to_original_size(flatten_image):
    image = flatten_image.reshape(-1,1)
    return image.reshape(IMAGE_SIZE)

if __name__ == "__main__":
    test_dish = "dish_1573234760"
    images = get_dishes_rgb_image()
    print(images[test_dish].shape)
    print("Number of dishes:", len(images))
    images_df = convert_dishes_rgb_image_to_dataframe(images)
    print(images_df)
    print(images_df.loc[images_df['dish_id'] == test_dish])


