import torch
from torch.optim import Adam
from tqdm import tqdm
from pathlib import Path
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_Nutrition5K_loaders,
    MultiTaskLoss,
    validate,
)
from model import BaseModel
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import MLSG


LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 1
NUM_WORKERS = 2
IMAGE_HEIGHT = 640
IMAGE_WIDTH = 640
PIN_MEMORY = True
LOAD_MODEL = False


def train(
    loader,
    model,
    optimizer,
    loss_fn,
):
    loop = tqdm(loader)
    total_loss = 0
    total_samples = 0

    for _, (images, nutrient_targets) in enumerate(loop):
        images = images.to(device=DEVICE)
        nutrient_targets = nutrient_targets.to(device=DEVICE)
        predictions = model(images)
        loss = loss_fn(predictions, nutrient_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix({"Multi loss": loss.item()})
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
    avg_loss = total_loss / total_samples
    return avg_loss


transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_HEIGHT),
        A.PadIfNeeded(
            min_height=IMAGE_HEIGHT,
            min_width=IMAGE_WIDTH,
            border_mode=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
            position="center",
        ),
        # A.Rotate(limit=35, p=1.0),
        A.Normalize(),
        ToTensorV2(),
    ]
)


base_dir = Path(r"E:\MLSingapore\MLSingapore\data\external\nutrition5k_dataset")

image_dir = base_dir / "imagery" / "realsense_overhead"
nutrition_dir = base_dir / "metadata" / "dish_metadata_cafe1.csv"

base_model = BaseModel().to(device=DEVICE)
optimizer = Adam(base_model.parameters(), lr=LEARNING_RATE)
nutrient_train_loss = MultiTaskLoss(validate=False).to(device=DEVICE)
nutrient_validation_loss = MultiTaskLoss(validate=True).to(device=DEVICE)


def main():
    train_loader, val_loader = get_Nutrition5K_loaders(
        image_dir=image_dir,
        nutrition_dir=nutrition_dir,
        foodsg_image_dir=None,
        foodsg_nutrition_dir=None,
        batch_size=BATCH_SIZE,
        transform=transforms,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    if LOAD_MODEL:
        load_checkpoint(filename="", model=base_model)

    base_model.train()

    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1} of {NUM_EPOCHS}:\n")
        train_loss = train(
            loader=train_loader,
            model=base_model,
            optimizer=optimizer,
            loss_fn=nutrient_train_loss,
        )
        # save model
        checkpoint = {
            "state_dict": base_model.state_dict(),
            "optimiser": optimizer.state_dict(),
        }
        save_checkpoint(state=checkpoint)

        print("Getting validation loss")
        val_loss = validate(
            model=base_model,
            data_loader=val_loader,
            loss_fn=nutrient_validation_loss,
            device=DEVICE,
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss vs epochs")
    plt.legend()
    plt.show()


def run_mlsg_validation():

    mlsg_base_dir = Path(
        r"E:\MLSingapore\MLSingapore\data\external\mlsg_validation\easy"
    )

    image_dir = mlsg_base_dir / "images"
    nutrition_dir = mlsg_base_dir / "easy.csv"

    dataset = MLSG(
        image_dir=image_dir, nutrition_dir=nutrition_dir, transform=transforms
    )

    test_loader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

    validate(
        model=base_model,
        data_loader=test_loader,
        loss_fn=nutrient_validation_loss,
        device=DEVICE,
        model_weights_path="",
    )


if __name__ == "__main__":
    main()
