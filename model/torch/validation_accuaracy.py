import torch 

def validation_accuaracy(model, device, validation_loader):
    """
    """
    v_corr = 0.0    
    model.eval()
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            preds = model(inputs)
            v_corr += torch.sum(preds.argmax(1) == labels)
            
        print('Accuracy: {:.4f}%'.format((v_corr / len(validation_loader.dataset)) * 100))


