import torch
from torch.utils.data import DataLoader
from dataset import Nutrition5K
import matplotlib.pyplot as plt
import numpy as np
import os


import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
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

def save_checkpoint(state, filename="base_model.pth"):
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
    nutrition_dir,
    batch_size,
    transform=None,
    num_workers=5,
    pin_memory=True,
    train_ratio=0.8,
):
    full_dataset = Nutrition5K(
        image_dir=image_dir,
        nutrition_dir=nutrition_dir,
        transform=transform,
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


def validate(model, data_loader, loss_fn, device="cuda"):
    model.eval()
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

    with torch.no_grad():
        for images, nutrient_targets in data_loader:
            images = images.to(device)
            nutrient_targets = nutrient_targets.to(device)
            losses = loss_fn(model(images), nutrient_targets).to(device)

            for key in loss_accumulator["error"]:
                loss_accumulator[key] += losses["error"][key]

            for key in loss_accumulator["actual"]:
                target_accumulator[key] += losses["actual"][key]

    num_batches = len(data_loader)
    mean_losses = {key: (value / num_batches) for key, value in loss_accumulator.items()}
    mean_targets = {key: (value / num_batches) for key, value in target_accumulator.items()}

    def get_percentage_loss(loss, target):
        return (loss / (target + 1e-6)) * 100
    
    calorie_percentage_loss = get_percentage_loss(loss=mean_losses["calories_mae"], target=mean_targets["true_calories"])
    mass_percentage_loss = get_percentage_loss(loss=mean_losses["mass_mae"], target=mean_targets["true_mass"])
    fats_percentage_loss = get_percentage_loss(loss=mean_losses["fats_mae"], target=mean_targets["true_fats"])
    carbs_percentage_loss = get_percentage_loss(loss=mean_losses["carbs_mae"], target=mean_targets["true_carbs"])
    proteins_percentage_loss = get_percentage_loss(loss=mean_losses["proteins_mae"], target=mean_targets["true_proteins"])

    result = f"\
    \nCalories MAE: {mean_losses["calories_mae"]} / {calorie_percentage_loss:.2f}\
    \nMass MAE: {mean_losses["mass_mae"]} / {mass_percentage_loss:.2f}\
    \nFats MAE: {mean_losses["fats_mae"]} / {fats_percentage_loss:.2f}\
    \nCarbohydrates MAE: {mean_losses["carbs_mae"]} / {carbs_percentage_loss:.2f}\
    \nProteins MAE: {mean_losses["proteins_mae"]} / {proteins_percentage_loss:.2f}
    "
    print(result)