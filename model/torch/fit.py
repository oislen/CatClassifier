import os
import torch 
import logging
from model.torch.validate import validate
from model.torch.ModelFit import ModelFit
from model.torch.EarlyStopper import EarlyStopper
from model.torch.checkpoints import save_checkpoint, load_checkpoint
from typing import Union
from beartype import beartype

@beartype
def fit(model, device:torch.device, criterion:torch.nn.CrossEntropyLoss, optimizer:torch.optim.SGD, train_dataloader:torch.utils.data.DataLoader, num_epochs:int=4, scheduler:Union[torch.optim.lr_scheduler.ReduceLROnPlateau,None]=None, valid_dataLoader:Union[torch.utils.data.DataLoader,None]=None, early_stopper:Union[EarlyStopper,None]=None, checkpoints_dir:Union[str,None]=None, load_epoch_checkpoint:Union[int,None]=None):
    """
    Fits model to specified data loader given the criterion and optimizer
    
    Parameters
    ----------
    model : CustomModelClass
        The customer torch model class object being fit
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
        The EarlyStopper object for halting fitting when performing validation
    checkpoints_dir : str
        The local folder location where model epoch checkpoints are to be read and wrote to, default is None
    load_epoch_checkpoint : int
        The epoch checkpoint to load and start from, default is None
    
    Returns
    -------
    """
    start_epoch = 1
    end_epoch = num_epochs + 1
    train_loss_list, train_acc_list, valid_loss_list, valid_acc_list = [], [], [], []
    model = model.to(device)
    n_total_steps = len(train_dataloader)
    # load previous model epoch checkpoint
    if (load_epoch_checkpoint != None) and (checkpoints_dir != None):
        checkpoints_filename = f"checkpoint_{model.model_id}_epoch_{load_epoch_checkpoint}.pt"
        checkpoint_filepath = os.path.join(checkpoints_dir, checkpoints_filename)
        checkpoint = load_checkpoint(filepath=checkpoint_filepath)
        # assign state dictionaries from checkpoint dictionary
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = load_epoch_checkpoint + 1
        train_loss_list, train_acc_list, valid_loss_list, valid_acc_list = checkpoint["train_loss_list"], checkpoint["train_acc_list"], checkpoint["valid_loss_list"], checkpoint["valid_acc_list"]
        logging.info(f"Read checkpoints from: {checkpoint_filepath}")
    for epoch in range(start_epoch, end_epoch):
        t_loss, t_corr = 0.0, 0.0
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):
            # load images and labels to device
            images = images.to(device)
            labels = labels.to(device)
            # forward pass
            preds = model.forward(images)
            loss = criterion(preds, labels)
            if scheduler != None:
                scheduler.step(loss)
            # backward and optimise
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # calculate metrics
            t_loss += loss.item() * images.size(0)
            t_corr += torch.sum(preds.argmax(1) == labels) 
            logging.info(f'Epoch [{epoch}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
        # update training loss and accuracy
        train_loss = t_loss / len(train_dataloader.dataset)
        train_acc = t_corr.item() / len(train_dataloader.dataset)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)  
        logging.info(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
        # calculate validation loss and accuracy if applicable
        if valid_dataLoader != None:
            valid_loss, valid_acc = validate(model=model, device=device, dataloader=valid_dataLoader, criterion=criterion)
            valid_loss_list.append(valid_loss)
            valid_acc_list.append(valid_acc)
            logging.info(f'Valid Loss: {loss.item():.4f}, Valid Accuracy: {valid_acc:.4f}')
            # if implementing early stopping
            if early_stopper != None and early_stopper.early_stop(valid_loss):
                logging.info(f"Applying early stopping criteria at validation loss: {valid_loss}")
                break
        # writing model checkpoints
        if (checkpoints_dir != None):
            checkpoints_filename = f"checkpoint_{model.model_id}_epoch_{epoch}.pt"
            checkpoint_filepath = os.path.join(checkpoints_dir, checkpoints_filename)
            checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), "train_loss_list":train_loss_list, "train_acc_list":train_acc_list, "valid_loss_list":valid_loss_list, "valid_acc_list":valid_acc_list}
            save_checkpoint(state=checkpoint, filepath=checkpoint_filepath)
            logging.info(f"Wrote checkpoints to: {checkpoint_filepath}")

    # create model fit object
    model_fit = ModelFit(loss=train_loss_list, accuracy=train_acc_list, val_loss=valid_loss_list, val_accuracy=valid_acc_list)
    return model, model_fit
