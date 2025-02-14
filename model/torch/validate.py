import torch 

def validate(model, device:torch.device, dataloader:torch.utils.data.DataLoader, criterion:torch.nn.CrossEntropyLoss) -> tuple:
    """Calculates validation loss and accuracy
    
    Parameters
    ----------
    model : CustomModelClass
        The customer torch model class object being fit
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
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        v_loss, v_corr = 0.0, 0.0
        for i, (images, labels) in enumerate(dataloader):
        #for i, (images, labels) in enumerate(zip(dataloader.dataset.image_tensors, dataloader.dataset.category_tensors)):
            # load images and labels to device
            images = images.to(device)
            labels = labels.to(device)
            # forward pass
            preds = model.forward(images)
            loss = criterion(preds, labels)
            # calculate metrics
            v_loss += loss.item() * images.size(0)
            v_corr += torch.sum(preds.argmax(1) == labels) 
        # update training loss and accuarcy
        valid_loss = v_loss / len(dataloader.dataset)
        valid_acc = v_corr.item() / len(dataloader.dataset)
    return (valid_loss, valid_acc)


