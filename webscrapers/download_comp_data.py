# import relevant libraries
import os
import subprocess
import zipfile
import logging
from beartype import beartype

@beartype
def download_comp_data(comp_name:str, data_dir:str, download_data:bool=True, unzip_data:bool=True, del_zip:bool=True):
    """Download Competition Data Documentation

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
    """
    
    logging.info('create zip file path ...')
    # define filenames
    zip_data_fname = '{}.zip'.format(comp_name)
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
        logging.info('data directory exists: {}'.format(data_dir))
    
    # if redownloading the data
    if download_data == True:
        logging.info('downing kaggle data ..')
        kaggle_cmd = 'kaggle competitions download --competition {} --path {} --force'.format(comp_name, data_dir)
        subprocess.run(kaggle_cmd.split())
    
    # if unzipping the data
    if unzip_data == True:
        if os.path.exists(zip_data_fpath) == False:
            raise OSError('file not found: {}'.format(zip_data_fpath))
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