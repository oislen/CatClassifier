import os
import platform
import logging
import pandas as pd 
import numpy as np
import random
from PIL import Image

# set huggingface hub directory
huggingface_hub_dir = 'E:\\huggingface'
if (platform.system() == 'Windows') and (os.path.exists(huggingface_hub_dir)):
    os.environ['TORCH_HOME'] = huggingface_hub_dir
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorflow.keras.preprocessing.image import load_img

# load custom scripts
import cons
from model.torch.VGG16_pretrained import VGG16_pretrained
from model.torch.AlexNet8 import AlexNet8
from model.torch.CustomDataset import CustomDataset
from model.torch.EarlyStopper import EarlyStopper
from model.utilities.plot_model import plot_model_fit
from model.utilities.plot_preds import plot_preds
from model.utilities.plot_image import plot_image
from model.utilities.plot_generator import plot_generator
from model.utilities.TimeIt import TimeIt
from model.utilities.commandline_interface import commandline_interface

# hyper-parameters
num_epochs = cons.min_epochs if cons.FAST_RUN else cons.max_epochs

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() and cons.check_gpu else 'cpu')

torch_transforms = transforms.Compose([
    transforms.Resize(size=[cons.IMAGE_WIDTH, cons.IMAGE_HEIGHT])  # resize the input image to a uniform size
    #,transforms.RandomRotation(30)
    #,transforms.RandomHorizontalFlip(p=0.05)
    #,transforms.RandomPerspective(distortion_scale=0.05, p=0.05)
    ,transforms.ToTensor()  # convert PIL Image or numpy.ndarray to tensor and normalize to somewhere between [0,1]
    ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # standardized processing
])

if __name__ == "__main__":
    
    # set up logging
    lgr = logging.getLogger()
    lgr.setLevel(logging.INFO)
    timeLogger = TimeIt()
    
    # handle input parameters
    input_params_dict = commandline_interface()

    if input_params_dict["run_model_training"]:

        logging.info("Generating dataframe of images...")
        # create a dataframe of filenames and categories
        filenames = os.listdir(cons.train_fdir)
        categories = [1 if filename.split('.')[0] == 'dog' else 0 for filename in filenames]
        df = pd.DataFrame({'filename': filenames, 'category': categories})
        frac = 0.05
        df = df.sample(frac = frac)
        df["categoryname"] = df["category"].replace(cons.category_mapper) 
        df['source'] = df['filename'].str.contains(pat='[cat|dog].[0-9]+.jpg', regex=True).map({True:'kaggle', False:'webscraper'})
        df["filepath"] = cons.train_fdir + '/' + df['filename']
        df["ndims"] = df['filepath'].apply(lambda x: len(np.array(Image.open(x)).shape))
        df = df.loc[df["ndims"] == 3, :].copy()
        timeLogger.logTime(parentKey="DataPrep", subKey="TrainDataLoad")
        
        logging.info("Plot sample image...")
        # random image plot
        sample = random.choice(filenames)
        image = load_img(os.path.join(cons.train_fdir, sample))
        plot_image(image, output_fpath=cons.torch_random_image_fpath, show_plot=False)
        timeLogger.logTime(parentKey="Plots", subKey="SampleImage")
        
        logging.info("Split into training, validation and test dataset...")
        # prepare data
        random_state = 42
        validate_df = df[df['source'] == 'kaggle'].sample(n=int(5000 * frac), random_state=random_state)
        train_df = df[~df.index.isin(validate_df.index)]
        train_df = train_df.reset_index(drop=True)
        validate_df = validate_df.reset_index(drop=True)
        timeLogger.logTime(parentKey="DataPrep", subKey="TrainValidationSplit")
        
        logging.info("Creating training and validation data loaders...")
        # set data constants
        total_train = train_df.shape[0]
        total_validate = validate_df.shape[0]
        # set train data loader
        train_dataset = CustomDataset(train_df, transforms=torch_transforms, mode='train')
        train_loader = DataLoader(train_dataset, batch_size=cons.batch_size, shuffle=True, num_workers=cons.num_workers, pin_memory=True)
        # set validation data loader
        validation_dataset = CustomDataset(train_df, transforms=torch_transforms, mode='train')
        validation_loader = DataLoader(validation_dataset, batch_size=cons.batch_size, shuffle=True, num_workers=cons.num_workers, pin_memory=True)
        timeLogger.logTime(parentKey="DataPrep", subKey="TrainValidationDataLoaders")
        
        logging.info("Plot example data loader images...")
        # datagen example
        example_generator = [(image.detach().numpy(), None) for images, labels in train_loader for image in images]
        plot_generator(generator=example_generator, mode='torch', output_fpath=cons.torch_generator_plot_fpath, show_plot=False)
        timeLogger.logTime(parentKey="Plots", subKey="DataLoader")
        
        logging.info("Initiate torch model...")
        # initiate cnn architecture
        #model = AlexNet8(num_classes=2).to(device)
        model = VGG16_pretrained(num_classes=2).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=cons.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='abs')
        early_stopper = EarlyStopper(patience=3, min_delta=0.3)
        timeLogger.logTime(parentKey="Modelling", subKey="InitiateTorchModel")
        
        logging.info("Fit torch model...")
        # fit torch model
        model.fit(device=device, criterion=criterion, optimizer=optimizer, train_dataloader=train_loader, num_epochs=num_epochs, scheduler=scheduler, valid_dataLoader=validation_loader, early_stopper=early_stopper)
        timeLogger.logTime(parentKey="Modelling", subKey="Fit")
        
        logging.info("Plot model fit results...")
        # plot model fits
        plot_model_fit(model_fit=model.model_fit, output_fdir=cons.torch_report_fdir, show_plot=False)
        timeLogger.logTime(parentKey="Plots", subKey="ModelFit")
        
        logging.info("Save fitted torch model to disk...")
        # save model
        model.save(output_fpath=cons.torch_model_pt_fpath)
        timeLogger.logTime(parentKey="ModelSerialisation", subKey="Write")

    if input_params_dict["run_testset_prediction"]:
        
        logging.info("Load fitted torch model from disk...")
        # load model
        #model = AlexNet8(num_classes=2).to(device)
        model = VGG16_pretrained(num_classes=2).to(device)
        model.load(input_fpath=cons.torch_model_pt_fpath)
        timeLogger.logTime(parentKey="ModelSerialisation", subKey="Load")
        
        logging.info("Generate test dataset...")
        # prepare test data
        test_filenames = os.listdir(cons.test_fdir)
        test_df = pd.DataFrame({'filename': test_filenames})
        test_df["filepath"] = cons.test_fdir + '/' + test_df['filename']
        test_df["idx"] = test_df['filename'].str.extract(pat='([0-9]+)').astype(int)
        test_df = test_df.set_index('idx').sort_index()
        nb_samples = test_df.shape[0]
        timeLogger.logTime(parentKey="TestSet", subKey="RawLoad")
        
        logging.info("Create test dataloader...")
        # set train data loader
        test_dataset = CustomDataset(test_df, transforms=torch_transforms, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=cons.batch_size, shuffle=False, num_workers=cons.num_workers, pin_memory=True)
        timeLogger.logTime(parentKey="TestSet", subKey="DataLoader")
        
        logging.info("Generate test set predictions...")
        # make test set predictions
        predict = model.predict(test_loader, device)
        test_df['category'] = np.argmax(predict, axis=-1)
        test_df["category"] = test_df["category"].replace(cons.category_mapper)
        timeLogger.logTime(parentKey="TestSet", subKey="ModelPredictions")
        
        logging.info("Plot example test set predictions...")
        # plot random sample predictions
        plot_preds(data=test_df, output_fpath=cons.torch_pred_images_fpath, show_plot=False)
        timeLogger.logTime(parentKey="Plots", subKey="TestSetPredictions")
        
        logging.info("Generate a sample submission file for kaggle...")
        # make submission
        submission_df = test_df.copy()
        submission_df['id'] = submission_df['filename'].str.split('.').str[0]
        submission_df['label'] = submission_df['category'].replace(cons.category_mapper)
        submission_df.to_csv(cons.submission_csv_fpath, index=False)
        timeLogger.logTime(parentKey="TestSet", subKey="SubmissionFile")