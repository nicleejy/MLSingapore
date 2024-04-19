import torch
import torch.nn as nn
from torchinfo import summary
import torchvision.models as models

# TODO: check encoder swap
# TODO: train Jeff yoloV8 classifier


class BaseModel(nn.Module):
    def __init__(self, input_height=640, input_width=640, custom_encoder=None):
        super(BaseModel, self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.pretrained_encoder = custom_encoder
        self.output_feature_size = 1024

        if custom_encoder is None:
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Dropout(0.3),
            )

        else:
            self.encoder = nn.Sequential(*list(custom_encoder.children())[:-1])

        self.fc1 = nn.Linear(
            (
                custom_encoder.fc.in_features
                if custom_encoder is not None
                else self.output_feature_size
            ),
            2048,
        )
        self.fc2 = nn.Linear(2048, 2048)

        self.fc_calories = nn.Linear(2048, 1)
        self.fc_mass = nn.Linear(2048, 1)
        self.fc_fat = nn.Linear(2048, 1)
        self.fc_carb = nn.Linear(2048, 1)
        self.fc_protein = nn.Linear(2048, 1)

        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        # calories, mass, fat, carb, protein
        calories = self.fc_calories(x)
        mass = self.fc_mass(x)
        fat = self.fc_fat(x)
        carb = self.fc_carb(x)
        protein = self.fc_protein(x)
        x = torch.cat((calories, mass, fat, carb, protein), dim=1)
        return x


# encoder = models.resnet50(weights="ResNet50_Weights.DEFAULT")

# model = BaseModel()
# batch_size = 16
# summary(model, input_size=(batch_size, 3, 640, 640))
