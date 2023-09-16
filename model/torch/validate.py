import torch 

def validate(model, device, dataloader, criterion):
    """
    """ 
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        v_loss, v_corr = 0.0, 0.0
        for i, (images, labels) in enumerate(dataloader):
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
        valid_acc = v_corr.cpu().numpy() / len(dataloader.dataset) * 100
        print(f'Valid Loss: {loss.item():.4f}, Valid Accuracy: {valid_acc:.4f}%')
    return valid_loss, valid_acc


