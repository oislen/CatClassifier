import torch 

def validate(model, device, validation_loader, criterion):
    """
    """
    valid_loss_list = []
    valid_acc_list = []  
    model.eval()
    with torch.no_grad():
        v_loss, v_corr = 0.0, 0.0
        for i, (images, labels) in enumerate(validation_loader):
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
        valid_loss = v_loss / len(validation_loader.dataset)
        valid_acc = v_corr.cpu().numpy() / len(validation_loader.dataset)
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)  
        print(f'Valid Loss: {loss.item():.4f} Accuracy: {valid_acc:.4f}%')
    return valid_loss_list, valid_acc_list


