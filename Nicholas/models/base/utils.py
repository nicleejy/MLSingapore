import torch
from torch.utils.data import DataLoader
from dataset import Nutrition5K
import os
import numpy as np
import torch

torch.manual_seed(0)

import torch.nn as nn
from PIL import Image


class MultiTaskLoss(nn.Module):
    """
    A custom PyTorch loss module for multi-task learning that combines three different
    losses: calories, mass, and macronutrients (fats, carbs, proteins) losses.

    Attributes:
        validate (bool): If set to True, the forward method will return a dictionary
                         containing individual losses and actuals for validation purposes.
                         Otherwise, it returns a single combined loss for training.

    Methods:
        forward(predictions, targets): Computes the combined loss or a dictionary of losses
                                       and actuals based on the 'validate' attribute.
    """

    def __init__(self, validate=False):
        super(MultiTaskLoss, self).__init__()
        self.validate = validate
        self.mae_loss = nn.L1Loss(reduction="mean")

    def forward(self, predictions, targets):
        # targets is of shape [N, 5] where:
        # targets[:, 0] - calories
        # targets[:, 1] - mass
        # targets[:, 2:5] - macronutrients (3 values - fats, carbs, proteins)

        # predicted values
        pred_calories = predictions[:, 0]
        pred_mass = predictions[:, 1]
        pred_fats = predictions[:, 2]
        pred_carbs = predictions[:, 3]
        pred_proteins = predictions[:, 4]

        pred_macronutrients = predictions[:, 2:5]

        # true values
        true_calories = targets[:, 0]
        true_mass = targets[:, 1]
        true_fats = targets[:, 2]
        true_carbs = targets[:, 3]
        true_proteins = targets[:, 4]

        true_macronutrients = targets[:, 2:5]

        loss_macros = self.mae_loss(pred_macronutrients, true_macronutrients)
        loss_calories = self.mae_loss(pred_calories, true_calories)
        loss_mass = self.mae_loss(pred_mass, true_mass)

        loss_fats = self.mae_loss(pred_fats, true_fats)
        loss_carbs = self.mae_loss(pred_carbs, true_carbs)
        loss_proteins = self.mae_loss(pred_proteins, true_proteins)

        combined_loss = loss_macros + loss_calories + loss_mass

        if not self.validate:
            return combined_loss

        batch_size = predictions.shape[0]
        # average error per dish for 1 batch
        # average true value per dish for 1 batch
        return {
            "combined": combined_loss.item(),
            "error": {
                "calories_mae": loss_calories.item(),
                "mass_mae": loss_mass.item(),
                "fats_mae": loss_fats.item(),
                "carbs_mae": loss_carbs.item(),
                "proteins_mae": loss_proteins.item(),
            },
            "actual": {
                "true_calories": true_calories.sum().item() / batch_size,
                "true_mass": true_mass.sum().item() / batch_size,
                "true_fats": true_fats.sum().item() / batch_size,
                "true_carbs": true_carbs.sum().item() / batch_size,
                "true_proteins": true_proteins.sum().item() / batch_size,
            },
        }


def get_percentage_loss(loss, target):
    return (loss / (target + 1e-6)) * 100


def save_checkpoint(state, filename="base_model.pth"):
    """
    Saves the model checkpoint to a file.

    Args:
        state (dict): State dictionary containing model weights and potentially other
                      parameters like optimizer state.
        filename (str): Desired path to save the checkpoint file. Defaults to "base_model.pth".
    """
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
    """
    Loads model weights from a specified file.

    Args:
        filename (str): Path to the checkpoint file to load.
        model (torch.nn.Module): Model instance on which state dictionary will be loaded.
    """
    print("Loading checkpoint...")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])
    # not required to save optimiser as using model for inference only


def get_Nutrition5K_loaders(
    image_dir,
    nutrition_dir,
    foodsg_image_dir,
    foodsg_nutrition_dir,
    batch_size,
    transform=None,
    num_workers=5,
    pin_memory=True,
    train_ratio=0.8,
    foodsg_dish_repetitions=20,
    dataset_ratio=1,
):
    """
    Prepares DataLoader instances for the Nutrition5K dataset.

    Args:
        image_dir (str): Directory path containing the images.
        nutrition_dir (str): Directory path containing the nutrition information.
        batch_size (int): Number of items per batch.
        transform (callable, optional): Transformations to apply to the images.
        num_workers (int): Number of worker processes to use for data loading.
        pin_memory (bool): Whether to use pinned memory.
        train_ratio (float): Proportion of dataset to use for training. The remaining portion is used for validation.

    Returns:
        tuple: A tuple containing the training DataLoader and validation DataLoader.
    """
    full_dataset = Nutrition5K(
        image_dir=image_dir,
        nutrition_dir=nutrition_dir,
        transform=transform,
        foodsg_image_dir=foodsg_image_dir,
        foodsg_nutrition_dir=foodsg_nutrition_dir,
        foodsg_dish_repetitions=foodsg_dish_repetitions,
        ratio=dataset_ratio,
    )
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def validate(model, data_loader, loss_fn, device="cuda", model_weights_path=None):
    """
    Validates the model using a given DataLoader and loss function, optionally loading model weights.

    Args:
        model (torch.nn.Module): The model to validate.
        data_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        loss_fn (callable): Loss function to use for validation (nn.L1Loss or MultiTaskLoss).
        device (str): Device to use for validation ('cuda' or 'cpu').
        model_weights_path (str, optional): Path to model weights to load before validation.

    Returns:
        str: Formatted string summarising the validation losses and percentage errors.
    """
    total_loss = 0
    loss_accumulator = {
        "calories_mae": 0,
        "mass_mae": 0,
        "fats_mae": 0,
        "carbs_mae": 0,
        "proteins_mae": 0,
    }
    target_accumulator = {
        "true_calories": 0,
        "true_mass": 0,
        "true_fats": 0,
        "true_carbs": 0,
        "true_proteins": 0,
    }

    if model_weights_path:
        load_checkpoint(
            filename=model_weights_path, model=model
        )  # load the weights into the model

    model.to(device=device)
    model.eval()

    with torch.no_grad():
        for images, nutrient_targets in data_loader:
            images = images.to(device)
            nutrient_targets = nutrient_targets.to(device)
            losses = loss_fn(model(images), nutrient_targets)

            for key in loss_accumulator:
                loss_accumulator[key] += losses["error"][key]

            for key in target_accumulator:
                target_accumulator[key] += losses["actual"][key]

            total_loss += losses["combined"]

    num_batches = len(data_loader)
    mean_losses = {
        key: (value / num_batches) for key, value in loss_accumulator.items()
    }
    mean_targets = {
        key: (value / num_batches) for key, value in target_accumulator.items()
    }

    mean_combined_loss = total_loss / num_batches

    calorie_percentage_loss = get_percentage_loss(
        loss=mean_losses["calories_mae"], target=mean_targets["true_calories"]
    )
    mass_percentage_loss = get_percentage_loss(
        loss=mean_losses["mass_mae"], target=mean_targets["true_mass"]
    )
    fats_percentage_loss = get_percentage_loss(
        loss=mean_losses["fats_mae"], target=mean_targets["true_fats"]
    )
    carbs_percentage_loss = get_percentage_loss(
        loss=mean_losses["carbs_mae"], target=mean_targets["true_carbs"]
    )
    proteins_percentage_loss = get_percentage_loss(
        loss=mean_losses["proteins_mae"], target=mean_targets["true_proteins"]
    )

    result = f"\
        \nCombined loss: {mean_combined_loss:.2f}\
        \nCalories MAE: {mean_losses['calories_mae']:.2f} / {calorie_percentage_loss:.2f}%\
        \nMass MAE: {mean_losses['mass_mae']:.2f} / {mass_percentage_loss:.2f}%\
        \nFats MAE: {mean_losses['fats_mae']:.2f} / {fats_percentage_loss:.2f}%\
        \nCarbohydrates MAE: {mean_losses['carbs_mae']:.2f} / {carbs_percentage_loss:.2f}%\
        \nProteins MAE: {mean_losses['proteins_mae']:.2f} / {proteins_percentage_loss:.2f}%"

    print(result)
    return mean_combined_loss


def predict(
    model,
    images,
    transforms,
    targets=None,
    device="cuda",
    model_weights_path="base_model.pth",
):
    """
    Predicts nutritional information for given images using a pre-trained model and evaluates the prediction error
    if targets are provided.

    This function takes either a single image or a list of images, applies the necessary transformations,
    and then uses the model to predict the nutritional values. It also loads the model weights from the specified path
    before prediction. If target nutritional values are provided, it will evaluate the percentage error for each nutrient.

    Args:
        model (torch.nn.Module): The model to use for predictions.
        images (str or list of str): Path(s) to the image(s) to be predicted.
        transforms (callable): A function that applies preprocessing transformations to the images.
        targets (list of dicts or dict, optional): Actual nutritional values to compare predictions against.
            For multiple images, provide a list of dictionaries with the same order as the images.
            For a single image, provide a single dictionary.
        device (str): Device to use for running predictions ('cuda' or 'cpu').
        model_weights_path (str): Path to the model weights file to load before prediction.

    Note:
        The function will print out the predicted nutritional values and, if targets are provided, the percentage error
        for each nutrient. Errors are calculated based on the absolute difference between predicted and actual values,
        relative to the actual value.
    """
    load_checkpoint(
        filename=model_weights_path, model=model
    )  # load the weights into the model
    model.to(device=device)
    model.eval()

    def infer(image_path):
        image = np.asarray(Image.open(image_path))
        transformed = transforms(image=image)
        transformed_image = transformed["image"].unsqueeze(0)
        with torch.no_grad():
            transformed_image = transformed_image.to(device=device)
            preds = model(transformed_image)
            preds = preds.cpu().numpy()[0]
            calories, mass, fats, carbs, proteins = (
                preds[0],
                preds[1],
                preds[2],
                preds[3],
                preds[4],
            )
            result = f"\
            \nImage {image_path}:\
            \nCalories - {calories:.2f}\
            \nMass - {mass:.2f}\
            \nFats - {fats:.2f}\
            \nCarbohydrates - {carbs:.2f}\
            \nProteins - {proteins:.2f}"
            print(result)
            return {
                "calories": calories,
                "mass": mass,
                "fats": fats,
                "carbs": carbs,
                "proteins": proteins,
            }

    def evaluate(target_dict, actual_dict):
        calorie_percentage_loss = get_percentage_loss(
            loss=abs(target_dict["calories"] - actual_dict["calories"]),
            target=target_dict["calories"],
        )
        mass_percentage_loss = get_percentage_loss(
            loss=abs(target_dict["mass"] - actual_dict["mass"]),
            target=target_dict["mass"],
        )
        fats_percentage_loss = get_percentage_loss(
            loss=abs(target_dict["fats"] - actual_dict["fats"]),
            target=target_dict["fats"],
        )
        carbs_percentage_loss = get_percentage_loss(
            loss=abs(target_dict["carbs"] - actual_dict["carbs"]),
            target=target_dict["carbs"],
        )
        proteins_percentage_loss = get_percentage_loss(
            loss=abs(target_dict["proteins"] - actual_dict["proteins"]),
            target=target_dict["proteins"],
        )
        result = f"\
        \nCalories Error %: {calorie_percentage_loss:.2f}%\
        \nMass Error %: {mass_percentage_loss:.2f}%\
        \nFats Error %: {fats_percentage_loss:.2f}%\
        \nCarbohydrates Error %: {carbs_percentage_loss:.2f}%\
        \nProteins Error %: {proteins_percentage_loss:.2f}%\n\n"
        print(result)

    if isinstance(images, list):
        for idx, image_path in enumerate(images):
            predicted_values = infer(image_path=image_path)
            if isinstance(targets, list) and len(targets) == len(images):
                expected_values = targets[idx]
                evaluate(target_dict=expected_values, actual_dict=predicted_values)

    elif isinstance(images, str):
        predicted_values = infer(image_path=images)
        if isinstance(targets, dict):
            evaluate(target_dict=targets, actual_dict=predicted_values)

    else:
        print("'images' accepts a file path or a list of file paths")
