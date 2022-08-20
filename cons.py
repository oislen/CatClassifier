import os
import sys

# set file directories
root_fdir = 'E:\\GitHub\\Cat-Classifier'
#root_fdir = '/home/ubuntu/Cat-Classifier'
data_fdir = os.path.join(root_fdir, 'data')
train_fdir = os.path.join(data_fdir, 'train')
test_fdir = os.path.join(data_fdir, 'test1')
model_fdir = os.path.join(root_fdir, 'model')
keras_fdir = os.path.join(root_fdir, 'keras')
kaggle_fdir = os.path.join(root_fdir, 'kaggle')
webscrapers_fdir_fdir = os.path.join(root_fdir, 'webscrapers')

# append directories to path
for fdir in [root_fdir, data_fdir, model_fdir, keras_fdir, kaggle_fdir, webscrapers_fdir_fdir]:
    sys.path.append(fdir)

# set kaggle competition name
comp_name = 'dogs-vs-cats'
download_data = False
unzip_data = True
del_zip = False

# set sample size
sample_size = 5000