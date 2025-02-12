# python model/arch/classify_image_torch.py --image_fpath E:/GitHub/CatClassifier/data/train/cat.0.jpg --model_fpath E:/GitHub/CatClassifier/data/models/VGG16.pt

import logging
import argparse
import platform
import os
import pandas as pd
import numpy as np
import sys
import re
from beartype import beartype

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

@beartype
def classify_image_torch(image_fpath:str, model_fpath:str=cons.torch_model_pt_fpath):
    """Classifies an input image using the torch model
    
    Parameters
    ----------
    image_fpath : str
        The full filepath to the image to classify using the torch model
    model_fpath : str
        The full filepath to the torch model to use for classification, default is cons.torch_model_pt_fpath
    
    Returns
    -------
    list
        The image file classification results as a recordset
    """
    
    logging.info("Loading torch model...")
    # load model
    #model = AlexNet8(num_classes=2).to(device)
    model = VGG16_pretrained(num_classes=2).to(device)
    model.load(input_fpath=model_fpath)
    
    logging.info("Generating dataset...")
    # prepare test data
    dataframe = pd.DataFrame({'filepath': [image_fpath]})
    
    logging.info("Creating dataloader...")
    # set train data loader
    dataset = CustomDataset(dataframe, transforms=torch_transforms, mode='test')
    loader = DataLoader(dataset, batch_size=cons.batch_size, shuffle=False, num_workers=cons.num_workers, pin_memory=True)
    
    logging.info("Classifying image...")
    # make test set predictions
    predict = model.predict(loader, device)
    dataframe['category'] = np.argmax(predict, axis=-1)
    dataframe["category"] = dataframe["category"].replace(cons.category_mapper)
    response = dataframe.to_dict(orient="records")
    logging.info(response)
    return response

if __name__ == "__main__":
    
    # set up logging
    lgr = logging.getLogger()
    lgr.setLevel(logging.INFO)
    
    # define argument parser object
    parser = argparse.ArgumentParser(description="Classify Image (Torch Model)")
    # add input arguments
    parser.add_argument("--image_fpath", action="store", dest="image_fpath", type=str, help="String, the full file path to the image to classify")
    parser.add_argument("--model_fpath", action="store", dest="model_fpath", type=str, default=cons.torch_model_pt_fpath, help="String, the full file path to the model to use for classification")
    # create an output dictionary to hold the results
    input_params_dict = {}
    # extract input arguments
    args = parser.parse_args()
    # classify image using torch model
    response = classify_image_torch(image_fpath=args.image_fpath, model_fpath=args.model_fpath)