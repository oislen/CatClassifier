import torch 

def fit_torch(model, device, criterion, optimizer, train_loader, num_epochs = 4):
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
            # origin shape: [4, 3, 32, 32] = 4, 3, 1024
            # input layer: 3 input channels, 6 output channels, 5 kernel size
            inputs = images.to(device)
            label = labels.to(device)

            # forward pass
            preds = model(inputs)
            loss = criterion(preds, label)

            # backward and optimise
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate metrics
            t_loss += loss.item() * inputs.size(0)
            t_corr += torch.sum(preds.argmax(1) == labels) 
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

        train_loss = t_loss / len(train_loader.dataset)
        train_acc = t_corr.cpu().numpy() / len(train_loader.dataset)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)  
        print('Train Loss: {:.4f} Accuracy: {:.4f}%'.format(train_loss, train_acc * 100))

    print('Finished Training')
    return model, train_loss_list, train_acc_list
