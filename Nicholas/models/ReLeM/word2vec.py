import pandas as pd
from pathlib import Path
import gensim
import json
import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



base_dir = Path(r"E:\MLSingapore\MLSingapore\data\external\Recipes5k")
recipe_dir = base_dir / "annotations" / "ingredients_simplified_Recipes5k.txt"



def get_ingredients_corpus(recipe_dir):
    df = pd.read_csv(recipe_dir, header=None, sep="\t", engine="python")
    ingredients_map = {"ingredients": []}
    for _, row in df.iterrows():
        ingr_list = row[0].split(",")
        ingredients_map["ingredients"].append(ingr_list)

    ingredients_df = pd.DataFrame(ingredients_map)
    ingredients_corpus = ingredients_df["ingredients"]
    return ingredients_corpus


def train_model(
    train_corpus, window=5, min_count=2, workers=4, epochs=10, path="./w2v_recipe.model"
):
    model = gensim.models.Word2Vec(
        window=window, min_count=min_count, workers=workers, vector_size=500
    )

    model.build_vocab(train_corpus, progress_per=500)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=epochs)
    model.save(path)
    print(f"w2v model saved to {path}")


def save_embeddings_as_json(
    model_path="./w2v_recipe.model", embeddings_path="./recipe_embeddings.json"
):
    model = gensim.models.Word2Vec.load(model_path)
    embeddings_dict = dict({})
    words = list(model.wv.index_to_key)
    for word in words:
        embeddings_dict[word] = model.wv.get_vector(word).tolist()
    embeddings_json = json.dumps(embeddings_dict, indent=4)
    # Writing to sample.json
    with open(embeddings_path, "w") as outfile:
        outfile.write(embeddings_json)

def load_embeddings(embeddings_path="./recipe_embeddings.json"):
    with open(embeddings_path) as f:
        recipe_embeddings_json = json.load(f)
    recipe_embeddings = {}
    for ingredient, embeddings_list in recipe_embeddings_json.items():
        recipe_embeddings[ingredient] = np.array(embeddings_list)
    return recipe_embeddings


def get_batch_mean_embeddings(recipes, recipe_embeddings, method="mean"):
    batch_embeddings = []
    for ingredient_str in recipes:
        ingredient_list = ingredient_str.split(",")
        valid_embeddings = [
            recipe_embeddings[ingredient]
            for ingredient in ingredient_list
            if ingredient in recipe_embeddings
        ]
        if valid_embeddings:
            if method == "mean":
                mean_embedding = np.mean(valid_embeddings, axis=0)
            elif method == "sum":
                mean_embedding = np.sum(valid_embeddings, axis=0)
        else:
            mean_embedding = np.zeros_like(list(recipe_embeddings.values())[0])
        batch_embeddings.append(mean_embedding)
    batch_embeddings_array = np.array(batch_embeddings)
    batch_embeddings_tensor = torch.tensor(batch_embeddings_array, dtype=torch.float32)
    return batch_embeddings_tensor


def visualise_recipe_embeddings(
    food_recipe_df, recipe_embeddings, method="mean", plot_dim=2
):
    batch_embeddings = []
    food_types = []

    for _, row in food_recipe_df.iterrows():
        image_url = row["image_url"]
        food_type = image_url.split("/")[0]
        food_types.append(food_type)

        ingredient_str = row["ingredients"]
        ingredient_list = ingredient_str.split(",")
        valid_embeddings = [
            recipe_embeddings[ingredient]
            for ingredient in ingredient_list
            if ingredient in recipe_embeddings
        ]
        if valid_embeddings:
            if method == "mean":
                recipe_embedding = np.mean(valid_embeddings, axis=0)
            elif method == "sum":
                recipe_embedding = np.sum(valid_embeddings, axis=0)
        else:
            recipe_embedding = np.zeros_like(list(recipe_embeddings.values())[0])
        batch_embeddings.append(recipe_embedding)

    batch_embeddings_array = np.array(batch_embeddings)

    pca = PCA(n_components=plot_dim)
    reduced_embeddings = pca.fit_transform(batch_embeddings_array)

    unique_food_types, food_type_indices = np.unique(food_types, return_inverse=True)
    cmap = plt.get_cmap("viridis", len(set(food_types)))

    if plot_dim == 2:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            c=food_type_indices,
            cmap=cmap,
            alpha=0.7,
        )
        cbar = plt.colorbar(scatter, ticks=range(len(set(food_types))))
        cbar.set_ticklabels(unique_food_types)
        plt.title("PCA of Recipe Embeddings (2D)")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.show()
    elif plot_dim == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            reduced_embeddings[:, 2],
            c=food_type_indices,
            cmap=cmap,
            alpha=0.7,
        )
        cbar = plt.colorbar(scatter, ticks=range(len(set(food_types))))
        cbar.set_ticklabels(unique_food_types)
        ax.set_title("PCA of Recipe Embeddings (3D)")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_zlabel("PC 3")
        plt.show()


def visualise_recipe_embeddings_labels(
    food_recipe_df, recipe_embeddings, method="mean", plot_dim=2
):
    batch_embeddings = []
    food_types = []

    for _, row in food_recipe_df.iterrows():
        image_url = row["image_url"]
        food_type = image_url.split("/")[0]
        food_types.append(food_type)

        ingredient_str = row["ingredients"]
        ingredient_list = ingredient_str.split(",")
        valid_embeddings = [
            recipe_embeddings[ingredient]
            for ingredient in ingredient_list
            if ingredient in recipe_embeddings
        ]
        if valid_embeddings:
            if method == "mean":
                recipe_embedding = np.mean(valid_embeddings, axis=0)
            elif method == "sum":
                recipe_embedding = np.sum(valid_embeddings, axis=0)
        else:
            recipe_embedding = np.zeros_like(list(recipe_embeddings.values())[0])
        batch_embeddings.append(recipe_embedding)

    batch_embeddings_array = np.array(batch_embeddings)

    pca = PCA(n_components=plot_dim)
    reduced_embeddings = pca.fit_transform(batch_embeddings_array)

    if plot_dim == 2:
        plt.figure(figsize=(10, 8))
        for i, (x, y) in enumerate(reduced_embeddings):
            plt.scatter(x, y, color="blue", alpha=0.7)
            plt.text(x, y, food_types[i], fontsize=7)
        plt.title("PCA of Recipe Embeddings (2D)")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.show()
    elif plot_dim == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        for i, (x, y, z) in enumerate(reduced_embeddings):
            ax.scatter(x, y, z, color="blue", alpha=0.7)
            ax.text(x, y, z, food_types[i], fontsize=7)
        ax.set_title("PCA of Recipe Embeddings (3D)")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_zlabel("PC 3")
        plt.show()
