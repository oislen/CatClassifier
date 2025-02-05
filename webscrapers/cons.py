import os
import re
import sys

# set root file directories
root_dir_re_match = re.findall(string=os.getcwd(), pattern="^.+CatClassifier")
root_fdir = root_dir_re_match[0] if len(root_dir_re_match) > 0 else os.path.join(".", "CatClassifier")
data_fdir = os.path.join(root_fdir, 'data')
creds_fdir = os.path.join(root_fdir, '.creds')
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
os.environ["KAGGLE_CONFIG_DIR"] = creds_fdir
comp_name = 'dogs-vs-cats'
download_data = True
unzip_data = True
del_zip = True

# webscraping constants
n_images = 6000
home_url = 'https://free-images.com'
output_dir =  os.path.join(data_fdir, '{search}')