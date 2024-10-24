import torch
import numpy as np

def predict(model, dataloader:torch.utils.data.DataLoader, device:torch.device) -> np.ndarray:
    """Predicts probabilities for a given data loader
    
    Parameters
    ----------
    model : CustomModelClass
        The customer torch model class object being fit
    dataloader : torch.utils.data.DataLoader
        The torch data loader to use when fitting the model
    device : torch.device
        The torch device to use when fitting the model
    
    Returns
    -------
    np.ndarry
        The dataloader target probabilities
    """
    fin_outputs = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model.forward(images)
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            proba = np.array(fin_outputs)
    return proba