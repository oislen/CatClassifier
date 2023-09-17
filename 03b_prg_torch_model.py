import os
import pandas as pd 
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorflow.keras.preprocessing.image import load_img

# load custom scripts
from model.torch.VGG16_pretrained import VGG16_pretrained
from model.torch.CustomDataset import CustomDataset
from model.plot_model import plot_model_fit
from model.torch.EarlyStopper import EarlyStopper
from data_prep.utilities.plot_preds import plot_preds
from data_prep.utilities.plot_image import plot_image
from data_prep.utilities.plot_generator import plot_generator
from data_prep.utilities.load_image import load_image
import cons

# create a dataframe of filenames and categories
filenames = os.listdir(cons.train_fdir)
categories = [1 if filename.split('.')[0] == 'dog' else 0 for filename in filenames]
df = pd.DataFrame({'filename': filenames, 'category': categories})
#frac = 0.05
#df = df.sample(frac = frac)
category_mapper = {0: 'cat', 1: 'dog'}
df["categoryname"] = df["category"].replace(category_mapper) 
df['source'] = df['filename'].str.contains(pat = '[cat|dog].[0-9]+\.jpg', regex = True).map({True:'kaggle', False:'webscraper'})
df["filepath"] = cons.train_fdir + '/' + df['filename']

# random image plot
sample = random.choice(filenames)
image = load_img(os.path.join(cons.train_fdir, sample))
plot_image(image, output_fpath = cons.random_image_fpath)

# prepare data
random_state = 42
validate_df = df[df['source'] == 'kaggle']#.sample(n = int(5000 * frac), random_state = random_state)
train_df = df[~df.index.isin(validate_df.index)]
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

# set data constants
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]

transform = transforms.Compose([
    transforms.Resize([cons.IMAGE_WIDTH, cons.IMAGE_HEIGHT])  # resize the input image to a uniform size
    ,transforms.RandomRotation(30)
    ,transforms.RandomHorizontalFlip(p=0.05)
    ,transforms.RandomPerspective(distortion_scale=0.05, p=0.05)
    ,transforms.ToTensor()  # convert PIL Image or numpy.ndarray to tensor and normalize to somewhere between [0,1]
    ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # standardized processing
])

# hyper-parameters
num_epochs = 25
batch_size = 64
learning_rate = 0.001

# set train datagen
train_dataset = CustomDataset(train_df, transform, mode = 'train')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# set validation datagen
validation_dataset = CustomDataset(train_df, transform, mode = 'train')
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

# datagen example
example_generator = [(image.detach().numpy(), None) for images, labels in train_loader for image in images]
plot_generator(generator = example_generator, mode = 'torch', output_fpath = cons.generator_plot_fpath)

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initiate cnn architecture
model = VGG16_pretrained(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='abs')
early_stopper = EarlyStopper(patience=3, min_delta=0.3)

# fit torch model
model.fit(device=device, criterion=criterion, optimizer=optimizer, train_dataloader=train_loader, num_epochs = num_epochs, scheduler=scheduler, valid_dataLoader=validation_loader, early_stopper=early_stopper)

# plot model fits
plot_model_fit(model_fit = model.model_fit, output_fdir = cons.report_fdir)

# save model
model.save(output_fpath=cons.torch_model_pt_fpath)

# load model
model = VGG16_pretrained(num_classes=2).to(device)
model.load(input_fpath=cons.torch_model_pt_fpath)

# prepare test data
test_filenames = os.listdir(cons.test_fdir)
test_df = pd.DataFrame({'filename': test_filenames})
test_df["filepath"] = cons.test_fdir + '/' + test_df['filename']
test_df["idx"] = test_df['filename'].str.extract(pat = '([0-9]+)').astype(int)
test_df = test_df.set_index('idx').sort_index()
nb_samples = test_df.shape[0]

# set train datagen
test_dataset = CustomDataset(test_df, transform, mode = 'test')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# make test set predictions
predict = model.predict(test_loader, device)
test_df['category'] = np.argmax(predict, axis=-1)
test_df["category"] = test_df["category"].replace(category_mapper) 

# plot random sample predictions
plot_preds(data = test_df, cons = cons, output_fpath = cons.pred_images_fpath)

# make submission
submission_df = test_df.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]
submission_df['label'] = submission_df['category'].replace({ 'dog': 1, 'cat': 0 })
submission_df.to_csv(cons.submission_csv_fpath, index=False)