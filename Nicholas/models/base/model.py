import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: check encoder swap
# TODO: train Jeff yoloV8 classifier


class BaseModel(nn.Module):
    def __init__(self, input_height=640, input_width=640, custom_encoder=None):
        super(BaseModel, self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.pretrained_encoder = custom_encoder

        if custom_encoder is None:
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.3),
            )
            feature_size = 128 * (input_height // 8) * (input_width // 8)
        else:
            self.encoder = custom_encoder
            # replace with actual output dimensions of the pretrained model
            feature_size = 128 * (input_height // 32) * (input_width // 32)

        self.fc1 = nn.Linear(feature_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 5)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # calories, mass, fat, carb, protein
        return x


# model = BaseModel()
# batch_size = 16
# summary(model, input_size=(batch_size, 3, 480, 640))
