from beartype import beartype
from torch import nn
from torchvision import models

from .BaseModel import base_model


class resnet_50_pretrained(base_model):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.model_id = "resnet_50_pretrained"
        self.resnet = models.resnet50(weights="DEFAULT")
        self.num_ftrs = self.resnet.fc.out_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.num_ftrs, out_features=num_classes)
        )

    @beartype
    def forward(self, x):
        """Applies a forward pass across an array x

        Parameters
        ----------
        x : array
            The array to apply a forward pass to

        Returns
        -------
        array
            The output array from the forward pass
        """
        x = self.resnet(x)
        x = self.classifier(x)
        return x
