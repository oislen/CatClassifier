import torch
import torch.nn as nn
import numpy as np

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

    def predict(self, dataloader, device):
        fin_outputs = []
        with torch.no_grad():
            for i, (images, labels) in enumerate(dataloader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = self.forward(images)
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
                proba = np.array(fin_outputs)
        return proba
    
    def save(self, output_fpath):
        torch.save(self.state_dict(), output_fpath)
        msg = f'Saved to {output_fpath}'
        return msg
    
    def load(self, input_fpath):
        self.load_state_dict(torch.load(input_fpath))
        self.eval()
        msg = f'Loaded from {input_fpath}'
        return msg