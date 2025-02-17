import os
import platform
import logging
import pandas as pd 
import numpy as np

# set huggingface hub directory
huggingface_hub_dir = 'E:\\huggingface'
if (platform.system() == 'Windows') and (os.path.exists(huggingface_hub_dir)):
    os.environ['TORCH_HOME'] = huggingface_hub_dir
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# load custom scripts
import cons
from model.torch.VGG16_pretrained import VGG16_pretrained
from model.torch.AlexNet8_pretrained import AlexNet8_pretrained
from model.torch.CustomDataset import CustomDataset
from model.torch.EarlyStopper import EarlyStopper
from model.utilities.plot_model import plot_model_fit
from model.utilities.plot_preds import plot_preds
from model.utilities.plot_image import plot_image
from model.utilities.plot_generator import plot_generator
from model.utilities.TimeIt import TimeIt
from model.utilities.commandline_interface import commandline_interface
from model.arch.load_image_v2 import load_image_v2, TorchLoadImages

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() and cons.check_gpu else 'cpu')

# initialise model
model = AlexNet8_pretrained(num_classes=2).to(device)
#model = VGG16_pretrained(num_classes=2).to(device)

random_state = 42

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
    
    logging.info("Parsing command line arguments...")
    # handle input parameters
    input_params_dict = commandline_interface()
    logging.info(input_params_dict)
    timeLogger.logTime(parentKey="Initialisation", subKey="CommandlineArguments")
    
    if input_params_dict["run_model_training"]:
        
        logging.info("Generating dataframe of images...")
        # load and shuffle the image file paths
        np.random.seed(random_state)
        image_filepaths=np.array([os.path.join(cons.train_fdir, x) for x in os.listdir(cons.train_fdir)])
        np.random.shuffle(image_filepaths)
        # create torch load images object
        sample_size = 20000
        torchLoadImages = TorchLoadImages(torch_transforms=torch_transforms, n_workers=None)
        df = pd.DataFrame.from_records(torchLoadImages.loadImages(image_filepaths[0:sample_size]))
        # only consider images with 3 dimensions
        df = df.loc[df["ndims"]==3, :]
        # flush data from memory
        del image_filepaths
        logging.info(f"df.shape: {df.shape}")
        timeLogger.logTime(parentKey="DataPrep", subKey="TrainDataLoad")
        
        logging.info("Plot sample image...")
        # random image plot
        plot_image(df['images'].values[1], output_fpath=cons.torch_random_image_fpath, show_plot=False)
        timeLogger.logTime(parentKey="Plots", subKey="SampleImage")
        
        logging.info("Plot example data loader images...")
        # data generator example
        plot_generator(generator=df['image_tensors'].values[:16].tolist(), mode='torch', output_fpath=cons.torch_generator_plot_fpath, show_plot=False)
        timeLogger.logTime(parentKey="Plots", subKey="DataLoader")
        
        logging.info("Split into training, validation and test dataset...")
        # prepare data
        validate_df = df.sample(frac=0.05, random_state=random_state, replace=False)
        train_df = df[~df.index.isin(validate_df.index)]
        train_df = train_df.reset_index(drop=True)
        validate_df = validate_df.reset_index(drop=True)
        logging.info(f"train_df.shape: {train_df.shape}")
        logging.info(f"validate_df.shape: {validate_df.shape}")
        timeLogger.logTime(parentKey="DataPrep", subKey="TrainValidationSplit")
        
        logging.info("Creating training and validation data loaders...")
        # set data constants
        total_train = train_df.shape[0]
        total_validate = validate_df.shape[0]
        # set train data loader
        train_dataset = CustomDataset(train_df)
        train_loader = DataLoader(train_dataset, batch_size=cons.batch_size, shuffle=True, num_workers=cons.num_workers, pin_memory=True, collate_fn=CustomDataset.collate_fn)
        # set validation data loader
        validation_dataset = CustomDataset(validate_df)
        validation_loader = DataLoader(validation_dataset, batch_size=cons.batch_size, shuffle=True, num_workers=cons.num_workers, pin_memory=True, collate_fn=CustomDataset.collate_fn)
        # flush data from memory
        del df
        del train_df
        del train_dataset
        del validate_df
        del validation_dataset
        timeLogger.logTime(parentKey="DataPrep", subKey="TrainValidationDataLoaders")
        
        logging.info("Initiate torch model...")
        logging.info(f"device: {device}")
        # initiate cnn architecture
        if device == "cuda":
            model = nn.DataParallel(model)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=cons.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='abs')
        early_stopper = EarlyStopper(patience=3, min_delta=0.3)
        timeLogger.logTime(parentKey="Modelling", subKey="InitiateTorchModel")
        
        logging.info("Fit torch model...")
        # hyper-parameters
        num_epochs = cons.min_epochs if cons.FAST_RUN else cons.max_epochs
        # fit torch model
        model.fit(device=device, criterion=criterion, optimizer=optimizer, train_dataloader=train_loader, num_epochs=num_epochs, scheduler=scheduler, valid_dataLoader=validation_loader, early_stopper=early_stopper, checkpoints_dir=cons.checkpoints_fdir, load_epoch_checkpoint=None)
        # flush data from memory
        del train_loader
        del validation_loader
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
        model.load(input_fpath=cons.torch_model_pt_fpath)
        timeLogger.logTime(parentKey="ModelSerialisation", subKey="Load")
        
        logging.info("Generate test dataset...")
        # create torch load images object
        torchLoadImages = TorchLoadImages(torch_transforms=torch_transforms, n_workers=None)
        test_df = pd.DataFrame.from_records(torchLoadImages.loadImages(filepaths=[os.path.join(cons.test_fdir, x) for x in os.listdir(cons.test_fdir)]))
        logging.info(f"test_df.shape: {test_df.shape}")
        timeLogger.logTime(parentKey="DataPrep", subKey="TrainDataLoad")
        
        logging.info("Create test dataloader...")
        # set train data loader
        test_dataset = CustomDataset(test_df)
        test_loader = DataLoader(test_dataset, batch_size=cons.batch_size, shuffle=False, num_workers=cons.num_workers, pin_memory=True, collate_fn=CustomDataset.collate_fn)
        timeLogger.logTime(parentKey="TestSet", subKey="DataLoader")
        
        logging.info("Generate test set predictions...")
        # make test set predictions
        predict = model.predict(test_loader, device)
        test_df['category'] = np.argmax(predict, axis=-1)
        test_df["category"] = test_df["category"].replace(cons.category_mapper)
        # flush data from memory
        del test_dataset
        del test_loader
        timeLogger.logTime(parentKey="TestSet", subKey="ModelPredictions")
        
        logging.info("Plot example test set predictions...")
        # plot random sample predictions
        plot_preds(data=test_df, output_fpath=cons.torch_pred_images_fpath, show_plot=False)
        timeLogger.logTime(parentKey="Plots", subKey="TestSetPredictions")
        
        logging.info("Generate a sample submission file for kaggle...")
        # make submission
        submission_df = test_df.copy()
        submission_df['id'] = submission_df['filenames'].str.split('.').str[0]
        submission_df['label'] = submission_df['category'].replace(cons.category_mapper)
        submission_df.to_csv(cons.submission_csv_fpath, index=False)
        # delete dataframes from memory
        del test_df
        del submission_df
        timeLogger.logTime(parentKey="TestSet", subKey="SubmissionFile")