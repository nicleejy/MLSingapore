from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from dataset import RecipeDataset
from utils import load_embeddings
from word2vec import visualise_recipe_embeddings, visualise_recipe_embeddings_labels


base_dir = Path(r"E:\MLSingapore\MLSingapore\data\external\Recipes5k")

image_dir = base_dir / "images"
recipe_dir = base_dir / "annotations" / "ingredients_simplified_Recipes5k.txt"

train_image_labels_dir = base_dir / "annotations" / "train_images.txt"
train_recipe_labels_dir = base_dir / "annotations" / "train_labels.txt"

train_ds = RecipeDataset(
    image_dir=image_dir,
    image_labels_dir=train_image_labels_dir,
    recipe_labels_dir=train_recipe_labels_dir,
    recipe_dir=recipe_dir,
    transform=None,
)

food_recipe_df = train_ds.__get_data__()
food_recipe_df = food_recipe_df[food_recipe_df["similarity"] == 1]

food_recipe_df = food_recipe_df.sample(n=500)


recipe_embeddings = load_embeddings(
    r"E:\MLSingapore\MLSingapore\recipe_embeddings.json"
)

visualise_recipe_embeddings_labels(
    food_recipe_df, recipe_embeddings, method="mean", plot_dim=3
)
