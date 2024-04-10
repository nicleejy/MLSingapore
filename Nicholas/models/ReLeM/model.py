import torch
from torch.optim import Adam
from transformers import BertModel, BertTokenizer
from resnet import ResNet50Encoder
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    visualise_transforms,
    get_cosine_loss,
    retrieve_closest_recipe,
    get_recipe_embeddings,
    get_ingredients_list,
    plot_embeddings_with_pca,
    check_accuracy,
)
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
import cv2
import matplotlib.pyplot as plt


LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_EPOCHS = 10
NUM_WORKERS = 8
IMAGE_HEIGHT = 300
IMAGE_WIDTH = 300
PIN_MEMORY = True
LOAD_MODEL = False


def train(loader, vision_encoder, text_encoder, tokenizer, optimizer, cosine_loss_fn):
    loop = tqdm(loader)
    total_loss = 0
    total_samples = 0
    for i, (images, ingredients) in enumerate(loop):
        images = images.to(device=DEVICE)
        # process texts to match BERT format
        inputs = tokenizer.batch_encode_plus(
            ingredients, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.to(device=DEVICE) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # generate text embeddings (detach to avoid gradient flow into BERT)
        with torch.no_grad():
            text_embeddings = text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).pooler_output

        # generate image embeddings
        image_embeddings = vision_encoder(images)
        targets = torch.ones(image_embeddings.shape[0], device=DEVICE)
        # calculate cosine similarity loss
        cosine_loss = cosine_loss_fn(
            image_embeddings, text_embeddings, targets
        )  # target similarity is 1 (perfect similarity)
        optimizer.zero_grad()
        cosine_loss.backward()
        optimizer.step()
        # update tqdm loop
        loop.set_postfix({"Cosine Loss": cosine_loss.item()})
        total_loss += cosine_loss.item() * images.size(0)
        total_samples += images.size(0)
    avg_loss = total_loss / total_samples
    return avg_loss


base_dir = Path(r"E:\MLSingapore\MLSingapore\data\external\Recipes5k")

image_dir = base_dir / "images"
recipe_dir = base_dir / "annotations" / "ingredients_simplified_Recipes5k.txt"

recipe_complex_dir = base_dir / "annotations" / "ingredients_Recipes5k.txt"

train_image_labels_dir = base_dir / "annotations" / "train_images.txt"
train_recipe_labels_dir = base_dir / "annotations" / "train_labels.txt"
val_image_labels_dir = base_dir / "annotations" / "val_images.txt"
val_recipe_labels_dir = base_dir / "annotations" / "val_labels.txt"
test_image_labels_dir = base_dir / "annotations" / "test_images.txt"
test_recipe_labels_dir = base_dir / "annotations" / "test_labels.txt"

ResNet_vision_encoder = ResNet50Encoder(pretrained=False).to(device=DEVICE)
# Pre-trained and fixed text encoder
BERT_text_encoder = BertModel.from_pretrained("bert-base-uncased").to(device=DEVICE)
BERT_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
optimizer = Adam(ResNet_vision_encoder.parameters(), lr=LEARNING_RATE)
cosine_loss_fn = nn.CosineEmbeddingLoss()

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
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ]
)


def main():

    for param in BERT_text_encoder.parameters():
        param.requires_grad = False

    train_loader, val_loader = get_loaders(
        image_dir=image_dir,
        recipe_dir=recipe_complex_dir,
        train_image_labels_dir=train_image_labels_dir,
        train_recipe_labels_dir=train_recipe_labels_dir,
        val_image_labels_dir=val_image_labels_dir,
        val_recipe_labels_dir=val_recipe_labels_dir,
        batch_size=BATCH_SIZE,
        train_transform=transforms,
        val_transform=transforms,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(filename="", model=ResNet_vision_encoder)

    ResNet_vision_encoder.train()
    BERT_text_encoder.eval()  # freeze text encoder

    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):

        print(f"Epoch {epoch + 1} of {NUM_EPOCHS}:\n")
        train_loss = train(
            loader=train_loader,
            vision_encoder=ResNet_vision_encoder,
            text_encoder=BERT_text_encoder,
            tokenizer=BERT_tokenizer,
            optimizer=optimizer,
            cosine_loss_fn=cosine_loss_fn,
        )
        # save model
        checkpoint = {
            "state_dict": ResNet_vision_encoder.state_dict(),
            "optimiser": optimizer.state_dict(),
        }
        save_checkpoint(state=checkpoint)
        # check accuracy

        print("Get validation loss")
        val_loss = get_cosine_loss(
            loader=val_loader,
            vision_encoder=ResNet_vision_encoder,
            text_encoder=BERT_text_encoder,
            tokenizer=BERT_tokenizer,
            device=DEVICE,
        )
        accuracy = check_accuracy(
            load_pretrained=False,
            vision_encoder=ResNet_vision_encoder,
            text_encoder=BERT_text_encoder,
            tokenizer=BERT_tokenizer,
            image_dir=image_dir,
            recipe_dir=recipe_complex_dir,
            transforms=transforms,
            image_labels_dir=val_image_labels_dir,
            recipe_labels_dir=val_recipe_labels_dir,
            device=DEVICE,
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.show()


def predict():
    test_image_path = r"E:\MLSingapore\MLSingapore\data\external\Recipes5k\images\spaghetti_carbonara\3_spaghetti_carbonara_hostedLargeUrl.jpg"
    vision_encoder_weights_path = r"E:\MLSingapore\MLSingapore\vision_encoder.pth"
    top_n_recipes, top_n_similarities = retrieve_closest_recipe(
        test_image_path=test_image_path,
        vision_encoder_weights_path=vision_encoder_weights_path,
        vision_encoder=ResNet_vision_encoder,
        text_encoder=BERT_text_encoder,
        tokenizer=BERT_tokenizer,
        recipe_dir=recipe_complex_dir,
        device=DEVICE,
        transforms=transforms,
    )
    for recipe in top_n_recipes:
        print(recipe)
    print(top_n_similarities)


def accuracy():
    vision_encoder_weights_path = r"E:\MLSingapore\MLSingapore\vision_encoder.pth"
    check_accuracy(
        load_pretrained=True,
        vision_encoder_weights_path=vision_encoder_weights_path,
        vision_encoder=ResNet_vision_encoder,
        text_encoder=BERT_text_encoder,
        tokenizer=BERT_tokenizer,
        image_dir=image_dir,
        recipe_dir=recipe_complex_dir,
        transforms=transforms,
        image_labels_dir=test_image_labels_dir,
        recipe_labels_dir=test_recipe_labels_dir,
    )


def visualise():

    ingredients_list = get_ingredients_list(recipe_dir=recipe_dir)
    recipe_embeddings_tensor = get_recipe_embeddings(
        ingredients_list, BERT_text_encoder, BERT_tokenizer, device="cuda"
    )
    plot_embeddings_with_pca(recipe_embeddings_tensor, ingredients_list)


if __name__ == "__main__":
    main()
    # predict()
    # visualise()
    # accuracy()
