import os
import sys
import re
import platform

# set huggingface hub directory
huggingface_hub_dir = 'E:\\huggingface'
if (platform.system() == 'Windows') and (os.path.exists(huggingface_hub_dir)):
    os.environ['TORCH_HOME'] = huggingface_hub_dir
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# set root file directories
root_dir_re_match = re.findall(string=os.getcwd(), pattern="^.+CatClassifier")
root_fdir = root_dir_re_match[0] if len(root_dir_re_match) > 0 else os.path.join(".", "CatClassifier")
data_fdir = os.path.join(root_fdir, 'data')
dataprep_fdir = os.path.join(root_fdir, 'data_prep')
env_fdir = os.path.join(root_fdir, 'environments')
model_fdir = os.path.join(root_fdir, 'model')
report_fdir = os.path.join(root_fdir, 'report')
keras_report_fdir = os.path.join(report_fdir, 'keras')
torch_report_fdir = os.path.join(report_fdir, 'torch')
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
keras_model_pickle_fpath = os.path.join(data_fdir, 'keras_model.keras')
torch_model_pt_fpath = os.path.join(data_fdir, 'torch_model.pt')
test_preds_pickle_fpath = os.path.join(data_fdir, 'test_preds.pickle')
submission_csv_fpath = os.path.join(data_fdir, 'submission.csv')
keras_random_image_fpath = os.path.join(keras_report_fdir, 'random_image.jpg')
keras_generator_plot_fpath = os.path.join(keras_report_fdir, 'generator_plot.jpg')
keras_pred_images_fpath = os.path.join(keras_report_fdir, 'pred_images.jpg')
torch_random_image_fpath = os.path.join(torch_report_fdir, 'random_image.jpg')
torch_generator_plot_fpath = os.path.join(torch_report_fdir, 'generator_plot.jpg')
torch_pred_images_fpath = os.path.join(torch_report_fdir, 'pred_images.jpg')

# set list containing all required directories
root_fdirs = [root_fdir, data_fdir,  dataprep_fdir, env_fdir, model_fdir, report_fdir, keras_report_fdir, torch_report_fdir, test_fdir, train_fdir, webscrapers_fdir]
sub_fdirs = [checkpoints_fdir, arch_fdir, utilities_fdir]

# append directories to path
for fdir in root_fdirs + sub_fdirs:
    sys.path.append(fdir)

# set sample size
train_sample_size = 1
test_sample_size = 1

FAST_RUN = False
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
batch_size=64
learning_rate=0.001
min_epochs=3
max_epochs=10
category_mapper={0: 'cat', 1: 'dog'}

# data generator constants
rescale = 1./255
rotation_range = 15
shear_range = 0.1
zoom_range = 0.2
horizontal_flip = True
width_shift_range = 0.1
height_shift_range = 0.1
shuffle = False

# multiprocessing
num_workers = os.environ.get("PARAM_NUM_WORKERS", os.cpu_count())
check_gpu = os.environ.get("PARAM_CHECK_GPU", False)