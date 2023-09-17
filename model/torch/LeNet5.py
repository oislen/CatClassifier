import torch
import torch.nn as nn
import numpy as np
# load custom modules
from model.torch.predict import predict as predict_module
from model.torch.save import save as save_module
from model.torch.load import load as load_module
from model.torch.fit import fit as fit_module
from model.torch.validate import validate as validate_module

class LeNet5(nn.Module):
    def __init__(self, num_classes=1000):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size=(5, 5), stride = (1, 1), padding = 'valid'),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = (5, 5), stride = (1, 1), padding = 'valid'),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features = 16*29*29, out_features = 120),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.25),
            nn.Linear(in_features = 120, out_features = 84),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.25),
            nn.Linear(in_features = 84, out_features = num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def fit(self, device, criterion, optimizer, train_dataloader, num_epochs = 4, scheduler = None, valid_dataLoader = None, early_stopping = False):
        self, self.model_fit = fit_module(self, device, criterion, optimizer, train_dataloader, num_epochs, scheduler, valid_dataLoader, early_stopping)

    def validate(self, device, dataloader, criterion):
        valid_loss, valid_acc = validate_module(self, device, dataloader, criterion)
        return valid_loss, valid_acc

    def predict(self, dataloader, device):
        proba = predict_module(self, dataloader, device)
        return proba
    
    def save(self, output_fpath):
        msg = save_module(self, output_fpath)
        return msg
    
    def load(self, input_fpath):
        msg = load_module(self, input_fpath)
        return msg