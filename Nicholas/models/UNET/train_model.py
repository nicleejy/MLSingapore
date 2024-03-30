import torch
import cv2
import albumentations as A
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from albumentations.pytorch.transforms import ToTensorV2
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 10
NUM_EPOCHS = 20
NUM_WORKERS = 8
IMAGE_MAX_SIZE = 400
IMAGE_HEIGHT = 400
IMAGE_WIDTH = 400
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = (
    r"E:\MLSingapore\MLSingapore\data\external\FoodSeg103\Images\img_dir\train"
)

VAL_IMG_DIR = r"E:\MLSingapore\MLSingapore\data\external\FoodSeg103\Images\img_dir\test"

TRAIN_MASK_DIR = (
    r"E:\MLSingapore\MLSingapore\data\external\FoodSeg103\Images\ann_dir\train"
)

VAL_MASK_DIR = (
    r"E:\MLSingapore\MLSingapore\data\external\FoodSeg103\Images\ann_dir\test"
)


def train(loader, model, optimiser, loss_fn, scaler):
    loop = tqdm(loader)
    for i, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.long().to(device=DEVICE)
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)  # [N, C, H, W]
            loss = loss_fn(predictions, targets)
        # backward
        optimiser.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():

    train_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_MAX_SIZE),
            A.PadIfNeeded(
                min_height=IMAGE_HEIGHT,
                min_width=IMAGE_WIDTH,
                border_mode=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            ),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )
    val_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_MAX_SIZE),
            A.PadIfNeeded(
                min_height=IMAGE_HEIGHT,
                min_width=IMAGE_WIDTH,
                border_mode=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            ),
            A.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )

    model = UNET(in_channels=3, out_channels=104).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    train_loader, val_loader = get_loaders(
        train_dir=TRAIN_IMG_DIR,
        train_maskdir=TRAIN_MASK_DIR,
        val_dir=VAL_IMG_DIR,
        val_maskdir=VAL_MASK_DIR,
        batch_size=BATCH_SIZE,
        train_transform=train_transforms,
        val_transform=val_transforms,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(filename="UNET_v1.pth.tar", model=model)

    model.train()

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1} of {NUM_EPOCHS}:\n")
        train(train_loader, model, optimiser, loss_fn, scaler)
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimiser": optimiser.state_dict(),
        }
        save_checkpoint(state=checkpoint, filename="UNET_v1.pth.tar")
        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)
        # save_predictions_as_imgs(
        #     val_loader, model, folder="saved_images", device=DEVICE
        # )


if __name__ == "__main__":
    main()
