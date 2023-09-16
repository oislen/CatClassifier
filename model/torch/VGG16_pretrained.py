from torch import nn
from torchvision import models
# load custom modules
from model.torch.predict import predict as predict_module
from model.torch.save import save as save_module
from model.torch.load import load as load_module

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
        proba = predict_module(self, dataloader, device)
        return proba
    
    def save(self, output_fpath):
        msg = save_module(self, output_fpath)
        return msg
    
    def load(self, input_fpath):
        msg = load_module(self, input_fpath)
        return msg