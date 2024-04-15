import torch
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary


class ResNet50Encoder(nn.Module):
    def __init__(self, output_features=500, pretrained=True):
        super(ResNet50Encoder, self).__init__()
        weights = "ResNet50_Weights.DEFAULT" if pretrained else None
        resnet50 = models.resnet50(weights=weights)
        self.features = nn.Sequential(*list(resnet50.children())[:-1])
        self.fc = nn.Linear(resnet50.fc.in_features, output_features)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# resnet50 = ResNet50Encoder()
# batch_size = 16
# summary(resnet50, input_size=(batch_size, 3, 300, 300))
