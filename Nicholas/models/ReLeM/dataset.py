import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import os
import random


class RecipeDataset(Dataset):

    def __init__(
        self,
        image_dir,
        image_labels_dir,
        recipe_labels_dir,
        recipe_dir,
        transform=None,
        pos_neg_split=1,
    ):
        self.image_dir = image_dir
        self.transform = transform
        positive_samples_df = self.__mergelabels(
            image_labels_dir, recipe_labels_dir, recipe_dir
        )
        negative_samples_df = self.__generate_negative_samples(positive_samples_df)
        self.data = self.__combine_dataframes(
            positive_samples_df, negative_samples_df, ratio=pos_neg_split
        )

    def __mergelabels(self, image_labels_dir, recipe_labels_dir, recipe_dir):
        # get image urls
        image_labels = pd.read_csv(
            image_labels_dir,
            sep=", ",
            engine="python",
            header=None,
        )
        # get corresponding ingredient indices (for image index 0, recipe is found at index 89)
        ingredient_labels = pd.read_csv(
            recipe_labels_dir,
            sep=", ",
            engine="python",
            header=None,
        )
        # get full ingredients list
        ingredients = pd.read_csv(recipe_dir, header=None, sep="\t", engine="python")
        # merge ingredient indices with ingredients
        ingredients_ordered = pd.merge(
            ingredient_labels, ingredients, left_on=0, right_index=True, how="left"
        )
        # combine image urls and ingredients
        data = pd.concat(
            [image_labels, ingredients_ordered[ingredients_ordered.columns[-1]]],
            sort=False,
            axis=1,
        )
        data.columns = ["image_url", "ingredients"]
        data["similarity"] = 1  # all positive examples
        return data

    def __generate_negative_samples(self, df):
        new_samples = {"image_url": [], "ingredients": []}
        food_types = [f for f in os.listdir(self.image_dir) if not f.startswith(".")]
        for _, row in df.iterrows():
            image_url, ingredients = row["image_url"], row["ingredients"]
            current_food_type = image_url.split("/")[0]
            neg_food_type = random.choice(food_types)
            while neg_food_type == current_food_type:
                neg_food_type = random.choice(food_types)
            neg_food_type_url = self.image_dir / neg_food_type
            neg_food = random.choice(
                [f for f in os.listdir(neg_food_type_url) if f.endswith(".jpg")]
            )
            neg_food_url = neg_food_type + "/" + neg_food
            new_samples["image_url"].append(neg_food_url)
            new_samples["ingredients"].append(ingredients)
        neg_df = pd.DataFrame(new_samples)
        neg_df["similarity"] = -1  # all negative examples
        return neg_df

    def __combine_dataframes(self, df1, df2, ratio=1):
        num_samples = round(min(len(df1), len(df2)) * ratio)
        df1_subset = df1.sample(n=num_samples)
        df2_subset = df2.sample(n=num_samples)
        combined_df = pd.concat([df1_subset, df2_subset], ignore_index=True, axis=0)
        return combined_df

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, index):
        img_path = self.image_dir / self.data.loc[index, "image_url"]
        image = np.asarray(Image.open(img_path))
        ingredients = self.data.loc[index, "ingredients"]
        target = self.data.loc[index, "similarity"]
        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]
            return image, ingredients, target


base_dir = Path(r"E:\MLSingapore\MLSingapore\data\external\Recipes5k")

image_dir = base_dir / "images"
recipe_dir = base_dir / "annotations" / "ingredients_simplified_Recipes5k.txt"

test_image_labels_dir = base_dir / "annotations" / "test_images.txt"
test_recipe_labels_dir = base_dir / "annotations" / "test_labels.txt"

dataset = RecipeDataset(
    image_dir=image_dir,
    image_labels_dir=test_image_labels_dir,
    recipe_dir=recipe_dir,
    recipe_labels_dir=test_recipe_labels_dir,
    pos_neg_split=1,
)
