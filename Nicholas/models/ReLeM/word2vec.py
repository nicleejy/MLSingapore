import pandas as pd
from pathlib import Path
import gensim
import json
import numpy as np
import torch


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


base_dir = Path(r"E:\MLSingapore\MLSingapore\data\external\Recipes5k")
recipe_dir = base_dir / "annotations" / "ingredients_simplified_Recipes5k.txt"

train_corpus = get_ingredients_corpus(recipe_dir=recipe_dir)

# train_model(train_corpus=train_corpus)
# save_embeddings_as_json(model_path="./w2v_recipe.model")
# model = gensim.models.Word2Vec.load(r"E:\MLSingapore\MLSingapore\w2v_recipe.model")
# print(model.wv.most_similar("oyster"))


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
            recipe_embeddings[ingredient] for ingredient in ingredient_list
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
