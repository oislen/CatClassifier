import os
import sys

# set root file directories
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

# set subdirectories
checkpoints_fdir = os.path.join(keras_fdir, 'checkpoints')
arch_fdir = os.path.join(keras_fdir, 'arch')

# set file paths
train_data_pickle_fpath = os.path.join(data_fdir, 'train_data.pickle')
test_data_pickle_fpath = os.path.join(data_fdir, 'test_data.pickle')
model_fit_pickle_fpath = os.path.join(report_fdir, 'model_fit.pickle')
keras_model_pickle_fpath = os.path.join(report_fdir, 'keras_model.h5')
test_preds_pickle_fpath = os.path.join(report_fdir, 'test_preds.pickle')

# set list containing all required directories
root_fdirs = [root_fdir, data_fdir,  dataprep_fdir, env_fdir, kaggle_fdir, keras_fdir, report_fdir, test_fdir, train_fdir, webscrapers_fdir]
sub_fdirs = [checkpoints_fdir, arch_fdir]

# append directories to path
for fdir in root_fdirs + sub_fdirs:
    sys.path.append(fdir)

# set kaggle competition name
comp_name = 'dogs-vs-cats'
download_data = False
unzip_data = True
del_zip = False

# set sample size
train_sample_size = 1
test_sample_size = 1