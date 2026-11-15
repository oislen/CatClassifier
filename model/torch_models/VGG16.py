import torch
from beartype import beartype
from torch import nn

from .BaseModel import base_model


class vgg_16(base_model):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.model_id = "vgg_16"
        self.features = nn.Sequential(
            # first convulation and pooling layer
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding="same",
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding="same",
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # second convulation and pooling layer
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding="same",
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding="same",
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # third convulation and pooling layer
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding="same",
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding="same",
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding="same",
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # fourth convulation and pooling layer
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding="same",
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding="same",
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding="same",
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # fifth convulation and pooling layer
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding="same",
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding="same",
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding="same",
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.classifier = nn.Sequential(
            # first dense layer with dropout regularization
            nn.Linear(in_features=512 * 4 * 4, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            # second dense layer with dropout regularization
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            # third dense layer for prediction
            nn.Linear(in_features=4096, out_features=num_classes),
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
