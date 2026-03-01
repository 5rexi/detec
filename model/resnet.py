import torch
import torch.nn as nn
import torchvision.models as models

class HeadHelmetResNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        self.backbone = models.resnet18(pretrained=pretrained)

        # 替换最后的 pooling + fc
        self.backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
