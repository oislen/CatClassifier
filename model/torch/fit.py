import torch 

def fit(model, device, criterion, optimizer, train_loader, num_epochs = 4, scheduler = None):
    """
    """
    train_loss_list = []
    train_acc_list = []
    model = model.to(device)
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        t_loss, t_corr = 0.0, 0.0
        model.train()
        for i, (images, labels) in enumerate(train_loader):
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
        train_loss = t_loss / len(train_loader.dataset)
        train_acc = t_corr.cpu().numpy() / len(train_loader.dataset)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)  
        print('Train Loss: {:.4f} Accuracy: {:.4f}%'.format(train_loss, train_acc * 100))

    print('Finished Training')
    return model, train_loss_list, train_acc_list
