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
    predict,
)
from model import BaseModel
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import MLSG
from torchvision import models

LEARNING_RATE = 0.0005
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 100
NUM_WORKERS = 8
IMAGE_HEIGHT = 400
IMAGE_WIDTH = 400
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
    for _, (images, nutrient_targets) in enumerate(loop):
        images = images.to(device=DEVICE)
        nutrient_targets = nutrient_targets.to(device=DEVICE)
        predictions = model(images)
        loss = loss_fn(predictions, nutrient_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix({"Multi loss": loss.item()})
        total_loss += loss.item()
    avg_loss_per_batch = total_loss / len(loader)  # loss of individual dish
    return avg_loss_per_batch


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
        # A.HorizontalFlip(p=0.5),
        # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        # A.ShiftScaleRotate(
        #     shift_limit=0.0625,
        #     scale_limit=0.2,
        #     rotate_limit=15,
        #     p=0.7,
        #     border_mode=cv2.BORDER_CONSTANT,
        #     value=(0, 0, 0),
        # ),
        A.Normalize(),
        ToTensorV2(),
    ]
)


base_dir = Path(r"E:\MLSingapore\MLSingapore\data\external\nutrition5k_dataset")

image_dir = base_dir / "imagery" / "realsense_overhead"
nutrition_dir = base_dir / "metadata" / "dish_metadata_cafe1.csv"

foodsg_base_dir = Path(r"E:\MLSingapore\MLSingapore\data\external\foodsg-233")

foodsg_image_dir = foodsg_base_dir / "images"
foodsg_nutrition_dir = foodsg_base_dir / "foodsg_233_metadata.csv"

encoder = models.resnet50(weights="ResNet50_Weights.DEFAULT")

base_model = BaseModel(
    input_height=IMAGE_HEIGHT, input_width=IMAGE_WIDTH, custom_encoder=encoder
).to(device=DEVICE)

optimizer = Adam(base_model.parameters(), lr=LEARNING_RATE)
nutrient_train_loss = MultiTaskLoss(validate=False).to(device=DEVICE)
nutrient_validation_loss = MultiTaskLoss(validate=True).to(device=DEVICE)


early_stopping_patience = 10
min_improvement = 3


def main():
    train_loader, val_loader = get_Nutrition5K_loaders(
        image_dir=image_dir,
        nutrition_dir=nutrition_dir,
        foodsg_image_dir=foodsg_image_dir,
        foodsg_nutrition_dir=foodsg_nutrition_dir,
        foodsg_dish_repetitions=15,
        dataset_ratio=0.5,
        batch_size=BATCH_SIZE,
        transform=transforms,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        train_ratio=0.8,
    )
    if LOAD_MODEL:
        load_checkpoint(filename="", model=base_model)

    base_model.train()

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1} of {NUM_EPOCHS}:\n")
        train_loss = train(
            loader=train_loader,
            model=base_model,
            optimizer=optimizer,
            loss_fn=nutrient_train_loss,
        )
        # save model
        # checkpoint = {
        #     "state_dict": base_model.state_dict(),
        #     "optimiser": optimizer.state_dict(),
        # }
        # save_checkpoint(state=checkpoint)

        print("Getting validation loss")
        val_loss = validate(
            model=base_model,
            data_loader=val_loader,
            loss_fn=nutrient_validation_loss,
            device=DEVICE,
        )
        if val_loss < best_val_loss - min_improvement:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            checkpoint = {
                "state_dict": base_model.state_dict(),
                "optimiser": optimizer.state_dict(),
            }
            print(f"Best model observed at epoch {epoch + 1}")
            save_checkpoint(
                state=checkpoint, filename=f"best_model_epoch{epoch + 1}.pth"
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"Stopping early at epoch {epoch + 1}. Validation loss has not improved for {early_stopping_patience} epochs."
            )
            break
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss vs epochs")
    plt.legend()
    plt.savefig("base_model.png")
    plt.show()


def run_mlsg_validation():

    mlsg_base_dir = Path(
        r"E:\MLSingapore\MLSingapore\data\external\mlsg_validation\hard"
    )

    image_dir = mlsg_base_dir / "images"
    nutrition_dir = mlsg_base_dir / "hard.csv"

    val_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_HEIGHT),
            A.PadIfNeeded(
                min_height=IMAGE_HEIGHT,
                min_width=IMAGE_WIDTH,
                border_mode=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
                position="center",
            ),
            A.Normalize(),
            ToTensorV2(),
        ]
    )

    dataset = MLSG(
        image_dir=image_dir, nutrition_dir=nutrition_dir, transform=val_transforms
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
        model_weights_path=r"E:\MLSingapore\MLSingapore\findings\base_food_sg\best_model_epoch13.pth",
    )


def run_predict():
    predict(
        model=base_model,
        images=[
            r"E:\MLSingapore\MLSingapore\data\external\nutrition5k_dataset\imagery\realsense_overhead\dish_1559239369\rgb.png",
            r"E:\MLSingapore\MLSingapore\data\external\nutrition5k_dataset\imagery\realsense_overhead\dish_1562873264\rgb.png",
            r"E:\MLSingapore\MLSingapore\data\external\mlsg_validation\easy\images\8.jpg",
        ],
        targets=[
            {
                "calories": 130.919998,
                "mass": 276,
                "fats": 0.612,
                "carbs": 31.919998,
                "proteins": 3.068000,
            },
            {
                "calories": 503.600220,
                "mass": 595,
                "fats": 35.514572,
                "carbs": 66.933792,
                "proteins": 33.333626,
            },
            {
                "calories": 548.88,
                "mass": 309,
                "fats": 23.88,
                "carbs": 59.08,
                "proteins": 24.43,
            },
        ],
        transforms=transforms,
        device=DEVICE,
        model_weights_path=r"E:\MLSingapore\MLSingapore\findings\base_food_sg\best_model_epoch13.pth",
    )


if __name__ == "__main__":
    main()
    # run_mlsg_validation()
    # run_predict()
