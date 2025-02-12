# python model/arch/classify_image_keras.py --image_fpath E:/GitHub/CatClassifier/data/train/cat.0.jpg --model_fpath E:/GitHub/CatClassifier/data/models/AlexNet8.keras

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

# load custom scripts
import cons

# load tensorflow / keras modules
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import load_model
from keras import optimizers

@beartype
def classify_image_keras(image_fpath:str, model_fpath:str=cons.keras_model_pickle_fpath):
    """Classifies an input image using the keras model
    
    Parameters
    ----------
    image_fpath : str
        The full filepath to the image to classify using the keras model
    model_fpath : str
        The full filepath to the keras model to use for classification, default is cons.keras_model_pickle_fpath
    
    Returns
    -------
    list
        The image file classification results as a recordset
    """
    
    logging.info("Loading keras model...")
    # load model
    model = load_model(model_fpath)
    
    logging.info("Generating dataset...")
    # prepare test data
    dataframe = pd.DataFrame({'filepath': [image_fpath]})
    
    logging.info("Creating dataloader...")
    # set data generator
    imagedatagenerator = ImageDataGenerator(rescale=cons.rescale)
    generator = imagedatagenerator.flow_from_dataframe(dataframe=dataframe, directory=cons.test_fdir, x_col='filepath', y_col=None, class_mode=None, target_size=cons.IMAGE_SIZE, batch_size=cons.batch_size, shuffle=cons.shuffle)

    logging.info("Classifying image...")
    # make test set predictions
    predict = model.predict(generator, steps=int(np.ceil(dataframe.shape[0]/cons.batch_size)))
    dataframe['category'] = np.argmax(predict, axis=-1)
    dataframe['category'] = dataframe['category'].replace(cons.category_mapper)
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
    parser.add_argument("--model_fpath", action="store", dest="model_fpath", type=str, default=cons.keras_model_pickle_fpath, help="String, the full file path to the model to use for classification")
    # create an output dictionary to hold the results
    input_params_dict = {}
    # extract input arguments
    args = parser.parse_args()
    # classify image using keras model
    response = classify_image_keras(image_fpath=args.image_fpath, model_fpath=args.model_fpath)