import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import os
import random


class Nutrition5K(Dataset):
    def __init__(
        self,
        image_dir,
        nutrition_dir,
        foodsg_image_dir=None,
        foodsg_nutrition_dir=None,
        transform=None,
        foodsg_dish_repetitions=20,
        ratio=1,
    ):
        self.image_dir = image_dir
        self.foodsg_image_dir = foodsg_image_dir
        self.data = self.__preprocess(nutrition_dir=nutrition_dir)
        if self.foodsg_image_dir is not None and foodsg_nutrition_dir is not None:
            foodsg_df = self.__process_foodsg(
                foodsg_nutrition_dir=foodsg_nutrition_dir,
                dish_repetitions=foodsg_dish_repetitions,
            )
            self.data = self.__combine_dataframes(
                df1=self.data, df2=foodsg_df, ratio=ratio
            )
            print(
                f"Merged Nutrition5K with FoodSG233 with {foodsg_dish_repetitions} samples per SG dish and {ratio * 100}% of each dataset (constrained by smaller dataset)"
            )
        num_rows = self.data.shape[0]
        print(f"Final dataset has {num_rows} training examples")
        self.transform = transform
        random.seed(10)

    def __get_image_path(self, dishID):
        return self.image_dir / dishID / "rgb.png"

    def __preprocess(self, nutrition_dir):
        df = pd.read_csv(
            nutrition_dir,
            on_bad_lines="warn",
            usecols=range(6),
            header=None,
            names=[
                "dishID",
                "total_calories",
                "total_mass",
                "total_fat",
                "total_carb",
                "total_protein",
            ],
        )
        df["image_path"] = df["dishID"].apply(self.__get_image_path)
        return df

    def __process_foodsg(self, foodsg_nutrition_dir, dish_repetitions):
        df = pd.read_csv(
            foodsg_nutrition_dir,
            on_bad_lines="warn",
            usecols=range(6),
        )
        df.rename(
            columns={
                "food": "dishID",
                "calories(kcal)": "total_calories",
                "mass(g)": "total_mass",
                "fat(g)": "total_fat",
                "carb(g)": "total_carb",
                "protein(g)": "total_protein",
            },
            inplace=True,
        )
        data = {"dishID": [], "image_path": []}
        for _, row in df.iterrows():
            dishname = row["dishID"]
            image_files = [
                image_file
                for image_file in os.listdir(self.foodsg_image_dir / dishname)
                if image_file.endswith(".jpg")
            ]
            if len(image_files) >= dish_repetitions:
                image_files_subset = random.sample(image_files, dish_repetitions)
            else:
                image_files_subset = image_files
            for image_file in image_files_subset:
                data["dishID"].append(dishname)
                data["image_path"].append(self.foodsg_image_dir / dishname / image_file)
        dishID_image_paths_df = pd.DataFrame(data)
        food_sg_df = dishID_image_paths_df.merge(df, on="dishID", how="left")
        return food_sg_df

    def __combine_dataframes(self, df1, df2, ratio=1):
        if ratio == 0:
            combined_df = pd.concat([df1, df2], ignore_index=True, axis=0)
            return combined_df
        num_samples = round(min(len(df1), len(df2)) * ratio)
        df1_subset = df1.sample(n=num_samples)
        df2_subset = df2.sample(n=num_samples)
        combined_df = pd.concat([df1_subset, df2_subset], ignore_index=True, axis=0)
        return combined_df

    def __len__(self):
        return len(self.data.index)

    # calories, mass, fat, carb, protein
    def __getitem__(self, index):
        img_path = self.image_dir / self.data.loc[index, "image_path"]
        image = np.asarray(Image.open(img_path))
        calories = self.data.loc[index, "total_calories"]
        mass = self.data.loc[index, "total_mass"]
        fat = self.data.loc[index, "total_fat"]
        carb = self.data.loc[index, "total_carb"]
        protein = self.data.loc[index, "total_protein"]
        nutrients = torch.tensor(
            [calories, mass, fat, carb, protein], dtype=torch.float32
        )
        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]
            return image, nutrients


# nutrition5k_base_dir = Path(
#     r"E:\MLSingapore\MLSingapore\data\external\nutrition5k_dataset"
# )

# image_dir = nutrition5k_base_dir / "imagery" / "realsense_overhead"
# nutrition_dir = nutrition5k_base_dir / "metadata" / "dish_metadata_cafe1.csv"

# foodsg_base_dir = Path(r"E:\MLSingapore\MLSingapore\data\external\foodsg-233")

# foodsg_nutrition_dir = foodsg_base_dir / "foodsg_233_metadata.csv"
# foodsg_image_dir = foodsg_base_dir / "images"

# transforms = A.Compose(
#     [
#         A.LongestMaxSize(max_size=640),
#         A.PadIfNeeded(
#             min_height=640,
#             min_width=640,
#             border_mode=cv2.BORDER_CONSTANT,
#             value=(0, 0, 0),
#             position="center",
#         ),
#         # A.Rotate(limit=35, p=1.0),
#         A.Normalize(),
#         ToTensorV2(),
#     ]
# )

# dataset = Nutrition5K(
#     image_dir=image_dir,
#     nutrition_dir=nutrition_dir,
#     transform=transforms,
#     foodsg_nutrition_dir=foodsg_nutrition_dir,
#     foodsg_image_dir=foodsg_image_dir,
#     ratio=1,
# )


class MLSG(Dataset):
    def __init__(
        self,
        image_dir,
        nutrition_dir,
        transform=None,
    ):
        self.image_dir = image_dir
        self.data = self.__preprocess(nutrition_dir)
        self.transform = transform

    def __get_image_path(self, imageID):
        image_name = str(imageID) + ".jpg"
        return self.image_dir / image_name

    def __preprocess(self, nutrition_dir):
        df = pd.read_csv(nutrition_dir)
        df.rename(
            columns={
                "food": "dishID",
                "calories(kcal)": "total_calories",
                "mass(g)": "total_mass",
                "fat(g)": "total_fat",
                "carb(g)": "total_carb",
                "protein(g)": "total_protein",
            },
            inplace=True,
        )
        df["image_path"] = df["image id"].apply(self.__get_image_path)
        df = df.drop("image id", axis=1)
        return df

    def __len__(self):
        return len(self.data.index)

    # calories, mass, fat, carb, protein
    def __getitem__(self, index):
        img_path = self.data.loc[index, "image_path"]
        image = np.asarray(Image.open(img_path))
        calories = self.data.loc[index, "total_calories"]
        mass = self.data.loc[index, "total_mass"]
        fat = self.data.loc[index, "total_fat"]
        carb = self.data.loc[index, "total_carb"]
        protein = self.data.loc[index, "total_protein"]
        nutrients = torch.tensor(
            [calories, mass, fat, carb, protein], dtype=torch.float32
        )
        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]
            return image, nutrients


# mlsg_base_dir = Path(r"E:\MLSingapore\MLSingapore\data\external\mlsg_validation\easy")

# image_dir = mlsg_base_dir / "images"
# nutrition_dir = mlsg_base_dir / "easy.csv"


# data = MLSG(image_dir=image_dir, nutrition_dir=nutrition_dir, transform=transforms)

# print(data.__getitem__(5))
