from torch import nn
import torch
from torchvision import models
import numpy as np

class VGG16_pretrained(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16_pretrained, self).__init__()
        self.resnet = models.vgg16(weights ="DEFAULT")
        self.num_ftrs = self.resnet.classifier[len(self.resnet.classifier)-1].out_features
        self.classifier = nn.Sequential(nn.Linear(in_features=self.num_ftrs, out_features=num_classes))

    def forward(self, x):
        x = self.resnet(x)
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