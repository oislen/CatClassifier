import os
import sys

# set file directories
root_fdir = 'E:\\GitHub\\cat_classifier'
data_fdir = os.path.join(root_fdir, 'data')
train_fdir = os.path.join(data_fdir, 'train')
test_fdir = os.path.join(data_fdir, 'test1')
scripts_fdir = os.path.join(root_fdir, 'scripts')
utilities_fdir = os.path.join(scripts_fdir, 'utilities')
keras_fdir = os.path.join(scripts_fdir, 'keras')
kaggle_fdir = os.path.join(scripts_fdir, 'kaggle')

# append directories to path
for fdir in [root_fdir, data_fdir, scripts_fdir, utilities_fdir, keras_fdir, kaggle_fdir]:
    sys.path.append(fdir)

# set kaggle competition name
comp_name = 'dogs-vs-cats'
download_data = True
unzip_data = True
del_zip = True

# set sample size
sample_size = 1000