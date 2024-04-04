import pandas as pd
import numpy as np
import os
import cv2

# Get the current directory of this file
CURR_DIR = os.path.dirname(__file__)

# Specify the path to the image data directory [Change to your own path]
DATA_DIR = "../../data/imagery/realsense_overhead"

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

if __name__ == "__main__":
    images = get_dishes_rgb_image()
    print("Number of dishes:", len(images))
    
