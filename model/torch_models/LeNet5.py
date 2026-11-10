import torch
from beartype import beartype
from torch import nn

from .BaseModel import base_model


class lenet_5(base_model):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.model_id = "lenet_5"
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=6,
                kernel_size=(5, 5),
                stride=(1, 1),
                padding="valid",
            ),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=(5, 5),
                stride=(1, 1),
                padding="valid",
            ),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=16 * 29 * 29, out_features=120),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=84, out_features=num_classes),
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
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
