import os
import sys
import platform

# set root file directories
root_fdir = 'E:\\GitHub\\Cat-Classifier' if platform.system() == 'Windows' else '/home/ubuntu/Cat-Classifier'
data_fdir = os.path.join(root_fdir, 'data')
dataprep_fdir = os.path.join(root_fdir, 'data_prep')
env_fdir = os.path.join(root_fdir, 'environments')
kaggle_fdir = os.path.join(root_fdir, 'kaggle')
model_fdir = os.path.join(root_fdir, 'model')
report_fdir = os.path.join(root_fdir, 'report')
test_fdir = os.path.join(data_fdir, 'test1')
train_fdir = os.path.join(data_fdir, 'train')
webscrapers_fdir = os.path.join(root_fdir, 'webscrapers')

# set subdirectories
checkpoints_fdir = os.path.join(data_fdir, 'checkpoints')
arch_fdir = os.path.join(model_fdir, 'arch')
utilities_fdir = os.path.join(dataprep_fdir, 'utilities')

# set file paths
train_data_pickle_fpath = os.path.join(data_fdir, 'train_data.pickle')
test_data_pickle_fpath = os.path.join(data_fdir, 'test_data.pickle')
model_fit_pickle_fpath = os.path.join(data_fdir, 'model_fit.pickle')
keras_model_pickle_fpath = os.path.join(data_fdir, 'keras_model.h5')
torch_model_pt_fpath = os.path.join(data_fdir, 'torch_model.pt')
test_preds_pickle_fpath = os.path.join(data_fdir, 'test_preds.pickle')
submission_csv_fpath = os.path.join(data_fdir, 'submission.csv')
random_image_fpath = os.path.join(report_fdir, 'random_image.jpg')
generator_plot_fpath = os.path.join(report_fdir, 'generator_plot.jpg')
pred_images_fpath = os.path.join(report_fdir, 'pred_images.jpg')

# set list containing all required directories
root_fdirs = [root_fdir, data_fdir,  dataprep_fdir, env_fdir, kaggle_fdir, model_fdir, report_fdir, test_fdir, train_fdir, webscrapers_fdir]
sub_fdirs = [checkpoints_fdir, arch_fdir, utilities_fdir]

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

# webscraping constants
n_images = 6000
home_url = 'https://free-images.com'
output_dir =  os.path.join(data_fdir, '{search}')

FAST_RUN = False
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
batch_size=16

# data generator constants
rescale = 1./255
rotation_range = 15
shear_range = 0.1
zoom_range = 0.2
horizontal_flip = True
width_shift_range = 0.1
height_shift_range = 0.1
shuffle = False