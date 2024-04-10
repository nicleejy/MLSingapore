import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd


class RecipeDataset(Dataset):

    def __init__(
        self, image_dir, image_labels_dir, recipe_labels_dir, recipe_dir, transform=None
    ):
        self.image_dir = image_dir
        self.transform = transform
        self.data = self.__mergelabels(image_labels_dir, recipe_labels_dir, recipe_dir)

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
        return data

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, index):
        img_path = self.image_dir / self.data.loc[index, "image_url"]
        image = np.asarray(Image.open(img_path))
        ingredients = self.data.loc[index, "ingredients"]
        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]
            return image, ingredients
