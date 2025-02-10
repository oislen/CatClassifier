# python arch/classify_image_torch.py --image_fpath E:/GitHub/CatClassifier/data/train/cat.0.jpg

import logging
import argparse
import platform
import os
import pandas as pd
import numpy as np
import sys
import re

# set root file directories
root_dir_re_match = re.findall(string=os.getcwd(), pattern="^.+CatClassifier")
root_fdir = root_dir_re_match[0] if len(root_dir_re_match) > 0 else os.path.join(".", "CatClassifier")
model_fdir = os.path.join(root_fdir, 'model')
sys.path.append(model_fdir)

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
from model.torch.CustomDataset import CustomDataset

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    # define argument parser object
    parser = argparse.ArgumentParser(description="Classify Image (Torch Model)")
    # add input arguments
    parser.add_argument("--image_fpath", action="store", dest="image_fpath", type=str, help="String, the full file path to the image to classify")
    # create an output dictionary to hold the results
    input_params_dict = {}
    # extract input arguments
    args = parser.parse_args()
    # map input arguments into output dictionary
    input_params_dict["image_fpath"] = args.image_fpath

    logging.info("Loading torch model...")
    # load model
    #model = AlexNet8(num_classes=2).to(device)
    model = VGG16_pretrained(num_classes=2).to(device)
    model.load(input_fpath=cons.torch_model_pt_fpath)

    logging.info("Generating dataset...")
    # prepare test data
    filenames = os.listdir(cons.test_fdir)
    df = pd.DataFrame({'filepath': [input_params_dict["image_fpath"]]})

    logging.info("Creating dataloader...")
    # set train data loader
    dataset = CustomDataset(df, transforms=torch_transforms, mode='test')
    loader = DataLoader(dataset, batch_size=cons.batch_size, shuffle=False, num_workers=cons.num_workers, pin_memory=True)

    logging.info("Classifying image...")
    # make test set predictions
    predict = model.predict(loader, device)
    df['category'] = np.argmax(predict, axis=-1)
    df["category"] = df["category"].replace(cons.category_mapper)
    logging.info(df.to_dict(orient="records"))
