import os
import pandas as pd 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# load custom scripts
from model.torch.AlexNet8 import AlexNet8
from model.torch.LeNet5 import LeNet5
from model.torch.VGG16 import VGG16
from model.torch.CustomDataset import CustomDataset
from model.torch.fit_torch import fit_torch
from model.torch.validation_accuaracy import validation_accuaracy
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
    transforms.Resize([128, 128]),  # resize the input image to a uniform size
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

# initiate cnn architecture
model = AlexNet8(num_classes=2).to(device)
model = LeNet5(num_classes=2).to(device)
model = VGG16(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# fit torch model
model, train_loss_list, train_acc_list = fit_torch(model, device, criterion, optimizer, train_loader, num_epochs = 4)

# calculate validation accuracy
validation_accuaracy(model, device, validation_loader)
