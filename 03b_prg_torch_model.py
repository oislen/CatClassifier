import pickle
import os
import random
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision
from torchvision import datasets , transforms
from torchvision import models


# load custom scripts
from data_prep.utilities.plot_image import plot_image
from data_prep.utilities.plot_generator import plot_generator
from data_prep.utilities.plot_preds import plot_preds
from model.torch.AlexNet8 import AlexNet8
from model.torch.CNN import CNN
from model.torch.CustomDataset import CustomDataset
from model.plot_model import plot_model_fit
import cons

# create a dataframe of filenames and categories
filenames = os.listdir(cons.train_fdir)
categories = [1 if filename.split('.')[0] == 'dog' else 0 for filename in filenames]
df = pd.DataFrame({'filename': filenames, 'category': categories})
frac = 0.05
df = df.sample(frac = frac)
df["category"] = df["category"]#.replace({0: 'cat', 1: 'dog'}) 
df['source'] = df['filename'].str.contains(pat = '[cat|dog].[0-9]+\.jpg', regex = True).map({True:'kaggle', False:'webscraper'})
df["filepath"] = cons.train_fdir + '/' + df['filename']

# prepare data
validate_df = df[df['source'] == 'kaggle'].sample(n = int(5000 * frac), random_state = 42)
train_df = df[~df.index.isin(validate_df.index)]
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

# set data constants
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]

transform = transforms.Compose([
    transforms.Resize([224, 224]),  # resize the input image to a uniform size
    transforms.ToTensor(),  # convert PIL Image or numpy.ndarray to tensor and normalize to somewhere between [0,1]
    transforms.Normalize(   # standardized processing
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

# set train datagen
train_dataset = CustomDataset(train_df, transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# set validation datagen
validation_dataset = CustomDataset(train_df, transform)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=True)

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper-parameters
num_epochs = 4
batch_size = 4
learning_rate = 0.001

# initiate LeNet5 architecture
model = AlexNet8(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#model = CNN().to(device)

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

model, train_loss_list, train_acc_list = fit_torch(model, device, criterion, optimizer, train_loader, num_epochs = 4)

def training_plot(train_loss_list, train_acc_list):
    """
    """
    num_epochs = len(train_loss_list)
    plt.figure()
    plt.title('Train Loss and Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('')
    plt.plot(range(1, num_epochs+1), np.array(train_loss_list), color='blue', linestyle='-', label='Train_Loss')
    plt.plot(range(1, num_epochs+1), np.array(train_acc_list), color='red', linestyle='-', label='Train_Accuracy')
    plt.legend()
    plt.show() 
    return 0

def validation_accuaracy():
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


