import torch
from torch.utils.data import DataLoader
from dataset import RecipeDataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.decomposition import PCA
from word2vec import get_batch_mean_embeddings, load_embeddings


def save_checkpoint(state, filename="vision_encoder.pth"):
    base_dir = os.path.dirname(filename)
    base_name = os.path.basename(filename)
    name, ext = os.path.splitext(base_name)
    counter = 1
    new_filename = filename
    while os.path.isfile(new_filename):
        new_filename = os.path.join(base_dir, f"{name}_{counter}{ext}")
        counter += 1
    # save the checkpoint to the new filename
    torch.save(state, new_filename)
    print(f"Checkpoint saved to {new_filename}")


def load_checkpoint(filename, model):
    print("Loading checkpoint...")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])
    # not required to save optimiser as using model for inference only


def get_loaders(
    image_dir,
    recipe_dir,
    train_image_labels_dir,
    train_recipe_labels_dir,
    val_image_labels_dir,
    val_recipe_labels_dir,
    batch_size,
    train_transform=None,
    val_transform=None,
    num_workers=5,
    pin_memory=True,
    pos_neg_split=1,
):
    train_ds = RecipeDataset(
        image_dir=image_dir,
        image_labels_dir=train_image_labels_dir,
        recipe_labels_dir=train_recipe_labels_dir,
        recipe_dir=recipe_dir,
        transform=train_transform,
        pos_neg_split=pos_neg_split,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = RecipeDataset(
        image_dir=image_dir,
        image_labels_dir=val_image_labels_dir,
        recipe_labels_dir=val_recipe_labels_dir,
        recipe_dir=recipe_dir,
        transform=val_transform,
        pos_neg_split=pos_neg_split,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def get_cosine_loss(loader, vision_encoder, device="cuda"):
    vision_encoder.eval()  # set vision encoder to evaluation mode

    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for images, ingredients, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            image_embeddings = vision_encoder(images)
            recipe_embeddings = load_embeddings()
            text_embeddings = get_batch_mean_embeddings(
                ingredients, recipe_embeddings=recipe_embeddings
            ).to(device)
            cosine_loss_fn = nn.CosineEmbeddingLoss(margin=0.2)
            loss = cosine_loss_fn(image_embeddings, text_embeddings, targets)
            total_loss += loss.item() * images.size(0)  # multiply by batch size
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    print(f"Validation Loss: {avg_loss}")
    return avg_loss


def get_ingredients_list(recipe_dir):
    ingredients = pd.read_csv(
        recipe_dir, header=None, names=["Recipe"], sep="\t", engine="python"
    )
    ingredients_list = ingredients.iloc[:, 0].tolist()
    return ingredients_list


def retrieve_closest_recipe(
    test_image_path,
    vision_encoder_weights_path,
    vision_encoder,
    recipe_dir,
    transforms,
    device="cuda",
    n=5,
):
    load_checkpoint(filename=vision_encoder_weights_path, model=vision_encoder)
    vision_encoder.eval()
    image = np.asarray(Image.open(test_image_path))
    augmentations = transforms(image=image)
    image = augmentations["image"]
    image = image.to(device=device)
    ingredients_list = get_ingredients_list(recipe_dir=recipe_dir)
    with torch.no_grad():
        image_embedding = vision_encoder(image.unsqueeze(0)).squeeze(0)
    recipe_embeddings = load_embeddings()
    ingredients_list = get_batch_mean_embeddings(
        ingredients_list, recipe_embeddings=recipe_embeddings
    ).to(device=device)
    similarities = cosine_similarity(
        image_embedding.cpu().numpy()[None, :], ingredients_list.cpu().numpy()
    ).flatten()
    # indices of the top n most similar recipes
    top_n_indices = np.argsort(similarities)[-n:][::-1]
    top_n_recipes = [ingredients_list[idx] for idx in top_n_indices]
    top_n_similarities = [similarities[idx] for idx in top_n_indices]
    return top_n_recipes, top_n_similarities


def get_data(recipe_dir, image_labels_dir, recipe_labels_dir):
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
    ingredients = pd.read_csv(
        recipe_dir, header=None, names=["Recipe"], sep="\t", engine="python"
    )
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


def visualise_transforms(image_dir, transforms, isNormalized=True):
    image = np.array(Image.open(image_dir))
    transformed_image = transforms(image=image)["image"]
    if isNormalized:
        transformed_image = transformed_image * 255
    im = Image.fromarray(transformed_image.astype(np.uint8))
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(im)
    plt.title("Transformed Image")
    plt.axis("off")
    plt.show()


def plot_embeddings_with_pca(embeddings_tensor, ingredients_list, size=200):
    embeddings = embeddings_tensor.cpu().numpy()
    size = min(size, len(ingredients_list))
    indices = np.random.choice(range(len(ingredients_list)), size=size, replace=False)
    subset_embeddings = embeddings_tensor[indices]
    subset_ingredients_list = [ingredients_list[i] for i in indices]

    embeddings = subset_embeddings.cpu().numpy()

    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    for i, label in enumerate(subset_ingredients_list):
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1])
        labels = str(label.split(",")[0:3])
        plt.text(
            reduced_embeddings[i, 0] + 0.03,
            reduced_embeddings[i, 1] + 0.03,
            labels,
            fontsize=5,
        )

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Recipe Embeddings Visualized with PCA")
    plt.show()
