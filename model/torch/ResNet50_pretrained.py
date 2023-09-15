from torch import nn
from torchvision import models

class ResNet50_pretrained(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet50_pretrained, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.num_ftrs = self.resnet.fc.out_features
        self.classifier = nn.Sequential(nn.Linear(in_features=self.num_ftrs, out_features=num_classes))

    def forward(self, x):
        x = self.resnet(x)
        x = self.classifier(x)
        return x