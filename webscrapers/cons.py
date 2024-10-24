import os
import sys
import platform

# set root file directories
root_fdir = 'E:\\GitHub\\Cat-Classifier' if platform.system() == 'Windows' else '/home/ubuntu/Cat-Classifier'
data_fdir = os.path.join(root_fdir, 'data')
dataprep_fdir = os.path.join(root_fdir, 'data_prep')
report_fdir = os.path.join(root_fdir, 'report')
test_fdir = os.path.join(data_fdir, 'test1')
train_fdir = os.path.join(data_fdir, 'train')
webscrapers_fdir = os.path.join(root_fdir, 'webscrapers')

# set list containing all required directories
root_fdirs = [root_fdir, data_fdir,  dataprep_fdir, report_fdir, test_fdir, train_fdir, webscrapers_fdir]

# append directories to path
for fdir in root_fdirs:
    sys.path.append(fdir)

# set kaggle competition name
comp_name = 'dogs-vs-cats'
download_data = False
unzip_data = True
del_zip = False

# webscraping constants
n_images = 6000
home_url = 'https://free-images.com'
output_dir =  os.path.join(data_fdir, '{search}')