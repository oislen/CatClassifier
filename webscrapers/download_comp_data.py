# import relevant libraries
import os
import subprocess
import zipfile
import logging

def download_comp_data(comp_name,
                       data_dir,
                       download_data = True, 
                       unzip_data = True, 
                       del_zip = True
                       ):
    
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
        Whether or not ti delete the zip file once data extraction is complete, default is True
    
    Returns
    -------
    """
    
    logging.info('checking inputs ...')
    
    # check for string data types
    str_types = [comp_name, data_dir]
    if any([type(str_inp) != str for str_inp in str_types]):
        raise TypeError('Input Type Error: the input parameters [comp_name, data_dir] must be string data types.')
        
    # check for boolean data types
    bool_types = [download_data, unzip_data, del_zip]
    if any([type(bool_inp) != bool for bool_inp in bool_types]):
        raise TypeError('Input Type Error: the input parameters [download_data, unzip_data, del_zip] must be boolean data types.')
    
    logging.info('create zip file path ...')
          
    # define filenames
    zip_data_fname = '{}.zip'.format(comp_name)
    
    # create file paths
    zip_data_fpath = os.path.join(data_dir, zip_data_fname)
    zip_train_fpath = os.path.join(data_dir, 'train.zip')
    zip_test_fpath = os.path.join(data_dir, 'test1.zip')
    
    logging.info('checking for data directory ...')
        
    # check data directory exists
    if os.path.exists(data_dir) == False:
        
        # create the directory
        os.makedirs(data_dir)
        
    # otherwise
    else:
        
        logging.info('data directory exists: {}'.format(data_dir))
    
    # if redownloading the data
    if download_data == True:
        
        logging.info('downing kaggle data ..')
        
        # define the kaggle api command to download the data
        kaggle_cmd = 'kaggle competitions download -c {} -p {}'.format(comp_name, data_dir)
        
        # run kaggle cmd in commandline
        subprocess.run(kaggle_cmd.split())
    
    # if unzipping the data
    if unzip_data == True:
    
        # check if zip file does not exists
        if os.path.exists(zip_data_fpath) == False:
            
            # raise os exception
            raise OSError('file not found: {}'.format(zip_data_fpath))
          
        # otherwise
        else:
            
            logging.info('unzipping data ...')
            
            # read zip file
            with zipfile.ZipFile(zip_data_fpath, "r") as zip_ref:
                
                # extract files
                zip_ref.extractall(data_dir)
            
            # read zip file
            with zipfile.ZipFile(zip_train_fpath, "r") as zip_ref:
                
                # extract files
                zip_ref.extractall(data_dir)
            
            # read zip file
            with zipfile.ZipFile(zip_test_fpath, "r") as zip_ref:
                
                # extract files
                zip_ref.extractall(data_dir)
    
    # if deleting zip file
    if del_zip == True:
        
        logging.info('deleting zip file ..')
        
        # delete zip file
        os.remove(path = zip_data_fpath)
        os.remove(path = zip_train_fpath)
        os.remove(path = zip_test_fpath)
