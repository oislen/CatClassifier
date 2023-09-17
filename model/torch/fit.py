import torch 
from model.torch.validate import validate
from model.torch.ModelFit import ModelFit

def fit(model, device, criterion, optimizer, train_dataloader, num_epochs = 4, scheduler = None, valid_dataLoader = None, early_stopper = None):
    """
    """
    train_loss_list, train_acc_list, valid_loss_list, valid_acc_list = [], [], [], []
    model = model.to(device)
    n_total_steps = len(train_dataloader)
    for epoch in range(num_epochs):
        t_loss, t_corr = 0.0, 0.0
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):
            # load images and labels to device
            images = images.to(device)
            label = labels.to(device)
            # forward pass
            preds = model.forward(images)
            loss = criterion(preds, label)
            if scheduler != None:
                scheduler.step(loss)
            # backward and optimise
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # calculate metrics
            t_loss += loss.item() * images.size(0)
            t_corr += torch.sum(preds.argmax(1) == labels) 
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
        # update training loss and accuarcy
        train_loss = t_loss / len(train_dataloader.dataset)
        train_acc = t_corr.cpu().numpy() / len(train_dataloader.dataset)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)  
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
        # calculate validation loss and accuracy if applicable
        if valid_dataLoader != None:
            valid_loss, valid_acc = validate(model=model, device=device, dataloader=valid_dataLoader, criterion=criterion)
            valid_loss_list.append(valid_loss)
            valid_acc_list.append(valid_acc)
            print(f'Valid Loss: {loss.item():.4f}, Valid Accuracy: {valid_acc:.4f}')
            # if implementing early stopping
            if early_stopper != None and early_stopper.early_stop(valid_loss):
                print(f"Applying early stopping criteria at validation loss: {valid_loss}")
                break
    # create model fit object
    model_fit = ModelFit(loss=train_loss_list, accuracy=train_acc_list, val_loss=valid_loss_list, val_accuracy=valid_acc_list)
    return model, model_fit
