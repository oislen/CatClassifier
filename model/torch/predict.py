import torch
import numpy as np

def predict(model, dataloader, device):
    fin_outputs = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model.forward(images)
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            proba = np.array(fin_outputs)
    return proba