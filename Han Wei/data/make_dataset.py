import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

# Get the current directory of this file
CURR_DIR = os.path.dirname(__file__)

# Specify the path to the image data directory [Change to your own path]
IMAGE_DATA_DIR = "../../data/imagery/realsense_overhead"
CSV_DATA_DIR = "../../data/metadata/nutrition5k_dataset_metadata_dish_metadata_cafe1.csv"

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
    for directory in os.listdir(IMAGE_DATA_DIR):
        directory_path = os.path.join(IMAGE_DATA_DIR, directory)

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

'''
Given an image dictionary, where
    key is dish_id
    value is RGB values
Return DataFrame with 
    first column - dish_id
    Remaining columns - calories, mass, fat, carb, protein
Where only dishes in image dictionary are kept in the dataframe

Note:
For the CSV, there exist rows that are formatted wrongly.
For these wrongly formatted rows, the following script will just skip the rows.
As such, there is a possibility where 
    length of dish_images > length of data returned
'''
def get_dish_nutrients(dish_images):
    columns = ['dish_id', 'calories', 'mass', 'fat', 'carb', 'protein']
    dishes = dish_images.keys()
    raw_data = pd.read_csv(CSV_DATA_DIR, on_bad_lines='skip', header=None, usecols=range(6))
    data = raw_data[raw_data[0].isin(dishes)]
    data.columns = columns
    return data

'''
Given images and nutrients dataframe,
Return a merged dataframe on dish_id
'''
def merge_images_nutrient(images_df, nutrients_df):
    nutrients_df = nutrients_df.merge(images_df, how='left', on='dish_id')
    return nutrients_df

'''
Given a DataFrame, with
    dish_id, mass, calories, fat, protein and carb columns
    image column as flatten image
Split the dataframe into feature and labels, where
    feature as the original shape of image
    label as the 5 nutrient values in the dataframe
Return a tuple (feature, label)
'''
def feature_label_split(data):
    original_feature = []
    features = data["image"].values
    for feature in features:
        feature = feature.reshape(-1,1)
        feature = feature.reshape(IMAGE_SIZE)
        original_feature.append(np.array(feature))
    labels = data[['mass', 'calories', 'fat', 'protein', 'carb']].values

    return np.array(original_feature), labels

'''
Given 
    train_test_ratio - ratio of data to use as train
    train_validate_ratio - ratio of data to use as train
    random_state - random seed
The function works through a pipeline
    1. Get images from DATA_DIR
    2. Convert the images into a Dataframe
    3. Get the nutrients values as Dataframe
    4. Merge the images and nutrients Dataframes
    5. Perform train test split
    6. Perform train validation split
    7. Return train, validate, test feature and labels
'''
def data_pipe(train_test_ratio, train_validate_ratio, random_state=0):
    print("Getting Images...")
    images = get_dishes_rgb_image()
    images_df = convert_dishes_rgb_image_to_dataframe(images)
    nutrients_df = get_dish_nutrients(images)
    print("Merging DataFrame...")
    data = merge_images_nutrient(images_df, nutrients_df)

    print("Splitting train, validate and test...")
    train, test = train_test_split(data, train_size=train_test_ratio, random_state=random_state)
    train, val = train_test_split(train, train_size=train_validate_ratio, random_state=random_state)

    return feature_label_split(train), feature_label_split(val), feature_label_split(test) 


if __name__ == "__main__":
    train, val, test = data_pipe(0.8, 0.9)
    X_train, y_train = train
    X_val, y_val = val
    X_test, y_test = test
    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_validate shape: ", X_val.shape)
    print("y_validate shape: ", y_val.shape)
    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape)


