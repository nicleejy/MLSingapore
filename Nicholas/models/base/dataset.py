import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2

class Nutrition5K(Dataset):
    def __init__(
        self,
        image_dir,
        foodsg_image_dir,
        nutrition_dir,
        foodsg_nutrition_dir=None,
        transform=None,
    ):
        self.image_dir = image_dir
        self.foodsg_image_dir = foodsg_image_dir
        self.data = self.__preprocess(nutrition_dir=nutrition_dir)
        self.__process_foodsg(foodsg_nutrition_dir=foodsg_nutrition_dir)
        self.transform = transform

    def __get_image_path(self, dishID):
        dishID = "dish_1556572657"
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
    

    def __process_foodsg(self, foodsg_nutrition_dir):
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


base_dir = Path(
    "/Users/nicholas/Documents/NUSDocuments/y3s2/CS3264/TermProject/MLSingapore/data/Nutrition5KSample"
)

image_dir = base_dir / "imagery" / "realsense_overhead"
nutrition_dir = base_dir / "metadata" / "dish_metadata_cafe1.csv"

foodsg_dir = Path(
    "/Users/nicholas/Documents/NUSDocuments/y3s2/CS3264/TermProject/MLSingapore/data/FoodSG/foodsg_233_metadata.csv"
)

transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=640),
        A.PadIfNeeded(
            min_height=640,
            min_width=640,
            border_mode=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
            position="center",
        ),
        # A.Rotate(limit=35, p=1.0),
        A.Normalize(),
        ToTensorV2(),
    ]
)

dataset = Nutrition5K(
    image_dir=image_dir,
    nutrition_dir=nutrition_dir,
    transform=transforms,
    foodsg_dir=foodsg_dir,
)
# image, nutrients = dataset.__getitem__(1)

# print(image.shape)
# print(nutrients.shape)
