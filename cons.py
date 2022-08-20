import os
import sys

# set file directories
root_fdir = 'E:\\GitHub\\Cat-Classifier'
#root_fdir = '/home/ubuntu/Cat-Classifier'
data_fdir = os.path.join(root_fdir, 'data')
dataprep_fdir = os.path.join(root_fdir, 'data_prep')
env_fdir = os.path.join(root_fdir, 'environments')
kaggle_fdir = os.path.join(root_fdir, 'kaggle')
keras_fdir = os.path.join(root_fdir, 'keras')
report_fdir = os.path.join(root_fdir, 'report')
test_fdir = os.path.join(data_fdir, 'test1')
train_fdir = os.path.join(data_fdir, 'train')
webscrapers_fdir = os.path.join(root_fdir, 'webscrapers')

# set list containing all required directories
fdirs = [root_fdir, data_fdir,  dataprep_fdir, env_fdir, kaggle_fdir, keras_fdir, report_fdir, test_fdir, train_fdir, webscrapers_fdir]

# append directories to path
for fdir in fdirs:
    sys.path.append(fdir)

# set kaggle competition name
comp_name = 'dogs-vs-cats'
download_data = False
unzip_data = True
del_zip = False

# set sample size
sample_size = 5000