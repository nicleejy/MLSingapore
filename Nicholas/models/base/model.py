import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self, input_height=480, input_width=640, custom_encoder=None):
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
            )
            feature_size = 128 * (input_height // 8) * (input_width // 8)
        else:
            self.encoder = custom_encoder
            # replace with actual output dimensions of the pretrained model
            feature_size = 128 * (input_height // 32) * (input_width // 32)

        self.fc1 = nn.Linear(feature_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Usage example with a custom pretrained encoder (dummy example, replace with real pretrained model)
# pretrained_model = SomePretrainedModel()
# model = BaseModel(pretrained_encoder=pretrained_model.encoder)


# model = BaseModel()
# batch_size = 16
# summary(model, input_size=(batch_size, 3, 480, 640))
