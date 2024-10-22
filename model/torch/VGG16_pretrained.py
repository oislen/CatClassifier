from torch import nn
from torchvision import models
# load custom modules
from model.torch.predict import predict as predict_module
from model.torch.save import save as save_module
from model.torch.load import load as load_module
from model.torch.fit import fit as fit_module
from model.torch.validate import validate as validate_module

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

    def fit(self, device, criterion, optimizer, train_dataloader, num_epochs = 4, scheduler = None, valid_dataLoader = None, early_stopper = None):
        self, self.model_fit = fit_module(self, device, criterion, optimizer, train_dataloader, num_epochs, scheduler, valid_dataLoader, early_stopper)

    def validate(self, device, dataloader, criterion):
        valid_loss, valid_acc = validate_module(self, device, dataloader, criterion)
        return valid_loss, valid_acc

    def predict(self, dataloader, device):
        proba = predict_module(self, dataloader, device)
        return proba
    
    def save(self, output_fpath):
        msg = save_module(self, output_fpath)
        return msg
    
    def load(self, input_fpath, weights_only=False):
        msg = load_module(self, input_fpath, weights_only=weights_only)
        return msg