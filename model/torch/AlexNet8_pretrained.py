import numpy as np
import torch
from torch import nn
from torchvision import models
# load custom modules
from model.torch.predict import predict as predict_module
from model.torch.save import save as save_module
from model.torch.load import load as load_module
from model.torch.fit import fit as fit_module
from model.torch.validate import validate as validate_module
from model.torch.EarlyStopper import EarlyStopper
from typing import Union
from beartype import beartype

class AlexNet8_pretrained(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet8_pretrained, self).__init__()
        self.model_id = "AlexNet8_pretrained"
        self.alexnet = models.alexnet(weights ="DEFAULT")
        self.num_ftrs = self.alexnet.classifier[len(self.alexnet.classifier)-1].out_features
        self.classifier = nn.Sequential(nn.Linear(in_features=self.num_ftrs, out_features=num_classes))

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
        x = self.alexnet(x)
        x = self.classifier(x)
        return x

    @beartype
    def fit(self, device:torch.device, criterion:torch.nn.CrossEntropyLoss, optimizer:torch.optim.SGD, train_dataloader:torch.utils.data.DataLoader, num_epochs:int=4, scheduler:Union[torch.optim.lr_scheduler.ReduceLROnPlateau,None]=None, valid_dataLoader:Union[torch.utils.data.DataLoader,None]=None, early_stopper:Union[EarlyStopper,None]=None, checkpoints_dir:Union[str,None]=None, load_epoch_checkpoint:Union[int,None]=None):
        """Fits model to specified data loader given the criterion and optimizer
        
        Parameters
        ----------
        device : torch.device
            The torch device to use when fitting the model
        criterion : torch.nn.CrossEntropyLoss
            The criterion to use when fitting the model
        optimizer : torch.optim.SGD
            The torch optimizer to use when fitting the model
        train_dataloader : torch.utils.data.DataLoader
            The torch data loader to use when fitting the model
        num_epochs : int
            The number of training epochs, default is 4
        scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau
            The torch scheduler to use when fitting the model, default is None
        valid_dataLoader : torch.utils.data.DataLoader
            The torch data loader to use for validation when fitting the model, default is None
        early_stopper : EarlyStopper
            The EarlyStopper object for halting fitting when performing validation, default is None
        checkpoints_dir : str
            The local folder location where model epoch checkpoints are to be read and wrote to, default is None
        load_epoch_checkpoint : int
            The epoch checkpoint to load and start from, default is None
        
        Returns
        -------
        """
        self, self.model_fit = fit_module(self, device, criterion, optimizer, train_dataloader, num_epochs, scheduler, valid_dataLoader, early_stopper, checkpoints_dir, load_epoch_checkpoint)

    @beartype
    def validate(self, device:torch.device, dataloader:torch.utils.data.DataLoader, criterion:torch.nn.CrossEntropyLoss) -> tuple:
        """Calculates validation loss and accuracy
        
        Parameters
        ----------
        device : torch.device
            The torch device to use when fitting the model
        dataloader : torch.utils.data.DataLoader
            The torch data loader to use when fitting the model
        criterion : torch.nn.CrossEntropyLoss
            The criterion to use when fitting the model
        
        Returns
        -------
        tuple
            The validation loss and accuracy
        """
        valid_loss, valid_acc = validate_module(self, device, dataloader, criterion)
        return (valid_loss, valid_acc)

    @beartype
    def predict(self, dataloader:torch.utils.data.DataLoader, device:torch.device) -> np.ndarray:
        """Predicts probabilities for a given data loader
        
        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            The torch data loader to use when fitting the model
        device : torch.device
            The torch device to use when fitting the model
        
        Returns
        -------
        np.ndarry
            The dataloader target probabilities
        """
        proba = predict_module(self, dataloader, device)
        return proba
    
    
    @beartype
    def save(self, output_fpath:str) -> str:
        """Writes a torch model to disk as a file
        
        Parameters
        ----------
        output_fpath : str
            The output file location to write the torch model to disk
        
        Returns
        -------
        str
            The model data message status
        """
        msg = save_module(self, output_fpath)
        return msg
    
    @beartype
    def load(self, input_fpath:str, weights_only:bool=False) -> str:
        """Loads a torch model from disk as a file
        
        Parameters
        ----------
        input_fpath : str
            The input file location to load the torch model from disk
        weights_only : bool
            Whether loading just the model weights or the full serialised model object, default is False
        
        Returns
        -------
        str
            The load model message status
        """
        msg = load_module(self, input_fpath, weights_only=weights_only)
        return msg