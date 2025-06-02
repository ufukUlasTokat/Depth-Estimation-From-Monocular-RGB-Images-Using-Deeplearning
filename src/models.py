import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet34, efficientnet_b0


class DepthRegressorEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = efficientnet_b0(pretrained=True)
        self.base.classifier = nn.Sequential(
            nn.Linear(self.base.classifier[1].in_features, 224 * 224),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.base(x)
        return x.view(-1, 1, 224, 224)
    
class DepthRegressorResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet34(pretrained=True)
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 224 * 224),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        return x.view(-1, 1, 224, 224)
    
class DepthRegressorResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet18(pretrained=True)
        self.base.fc = nn.Sequential(
            nn.Linear(self.base.fc.in_features, 224 * 224),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.base(x)
        x = x.view(-1, 1, 224, 224)
        return x
    
class DepthRegressorDenseNet(nn.Module):
    def __init__(self):
        super(DepthRegressorDenseNet, self).__init__()
        self.base = models.densenet121(pretrained=True)
        self.base.classifier = nn.Sequential(
            nn.Linear(self.base.classifier.in_features, 224 * 224),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.base(x)
        x = x.view(-1, 1, 224, 224)
        return x
    
class DepthRegressorMobileNet(nn.Module):
    def __init__(self):
        super(DepthRegressorMobileNet, self).__init__()
        self.base = models.mobilenet_v2(pretrained=True)
        self.base.classifier = nn.Sequential(
            nn.Linear(self.base.last_channel, 224 * 224),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.base(x)
        x = x.view(-1, 1, 224, 224)
        return x