import torch
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary


class ResNet50Encoder(nn.Module):
    def __init__(self, output_features=100, pretrained=True):
        super(ResNet50Encoder, self).__init__()
        weights = "ResNet50_Weights.DEFAULT" if pretrained else None
        resnet50 = models.resnet50(weights=weights)
        self.features = nn.Sequential(*list(resnet50.children())[:-1])
        in_features = resnet50.fc.in_features

        self.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, output_features),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# resnet50 = ResNet50Encoder()
# batch_size = 16
# summary(resnet50, input_size=(batch_size, 3, 300, 300))
