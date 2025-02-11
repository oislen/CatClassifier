# import relevant libraries
import os
import subprocess
import zipfile
import logging
from beartype import beartype

@beartype
def download_comp_data(
    comp_name:str, 
    data_dir:str, 
    download_data:bool=True, 
    unzip_data:bool=True, 
    del_zip:bool=True
    ):
    """Download Competition Data

    Parameters
    ----------
    
    comp_name : str
        The name of the competition to download data for.
    data_dir : str 
        The data directory to download and extract the data to.
    download_data : bool
        Whether or not to download the data, default is True.
    unzip_data : bool 
        Whether or not to unzip the data, default is True.
    del_zip : bool
        Whether or not to delete the zip file once data extraction is complete, default is True
    
    Returns
    -------

    Example
    -------
    download_comp_data(
        comp_name="dogs-vs-cats",
        data_dir="E:\\GitHub\\CatClassifier\\data", 
        download_data=True,
        unzip_data=True,
        del_zip=True
        )
    """
    
    logging.info('create zip file path ...')
    # define filenames
    zip_data_fname = f'{comp_name}.zip'
    # create file paths
    zip_data_fpath = os.path.join(data_dir, zip_data_fname)
    zip_train_fpath = os.path.join(data_dir, 'train.zip')
    zip_test_fpath = os.path.join(data_dir, 'test1.zip')
    # combine paths in a list
    zip_fpaths_list = [zip_data_fpath, zip_train_fpath, zip_test_fpath]
    
    logging.info('checking for data directory ...')
    # check data directory exists
    if os.path.exists(data_dir) == False:
        os.makedirs(data_dir)
    else:
        logging.info(f'data directory exists: {data_dir}')
    
    # if downloading the data
    if download_data == True:
        logging.info('downing kaggle data ..')
        kaggle_cmd = f'kaggle competitions download --competition {comp_name} --path {data_dir} --force'
        subprocess.run(kaggle_cmd.split())
    
    # if unzipping the data
    if unzip_data == True:
        if os.path.exists(zip_data_fpath) == False:
            raise OSError(f'file not found: {zip_data_fpath}')
        else:
            for zip_fpath in zip_fpaths_list:
                logging.info(f'unzipping data {zip_fpath} ...')
                with zipfile.ZipFile(zip_fpath, "r") as zip_ref:
                    zip_ref.extractall(data_dir)
    
    # if deleting zip file
    if del_zip == True:
        for zip_fpath in zip_fpaths_list:
            logging.info('deleting zip file {zip_fpath} ...')
            os.remove(path = zip_fpath)

@beartype
def download_models(
    model_instance_url:str, 
    model_dir:str
    ):
    """Download Kaggle Models

    Parameters
    ----------
    
    model_instance_url : str
        Model Instance Version URL suffix in format <owner>/<model-name>/<framework>/<instance-slug>/<version-number>.
    model_dir : str 
        Folder where file(s) will be downloaded.
    
    Returns
    -------

    Example
    -------
    download_models(
        model_instance_url="oislen/cat-classifier-cnn-models/pyTorch/default/1",
        model_dir="E:\\GitHub\\CatClassifier\\data\\models"
        )
    """

    logging.info('checking for data directory ...')
    # check data directory exists
    if os.path.exists(model_dir) == False:
        os.makedirs(model_dir)
    else:
        logging.info(f'model directory exists: {model_dir}')
    
    # downloading the model
    logging.info('downloading kaggle model ..')
    kaggle_cmd = f'kaggle models instances versions download --path {model_dir} --untar --force {model_instance_url}'
    subprocess.run(kaggle_cmd.split())
