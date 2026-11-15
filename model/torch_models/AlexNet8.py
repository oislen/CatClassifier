import torch
from beartype import beartype
from torch import nn

from .BaseModel import base_model


class alexnet_8(base_model):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.model_id = "alexnet_8"
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=96,
                kernel_size=(11, 11),
                stride=(4, 4),
                padding="valid",
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.Conv2d(
                in_channels=96,
                out_channels=256,
                kernel_size=(5, 5),
                stride=(1, 1),
                padding="same",
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.Conv2d(
                in_channels=256,
                out_channels=384,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding="same",
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=384,
                out_channels=384,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding="same",
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=384,
                out_channels=256,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding="same",
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2 * 2, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
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
