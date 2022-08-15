import os
import sys

root_fdir = 'E:\\GitHub\\cat_classifier'
data_fdir = os.path.join(root_fdir, 'data')
scripts_fdir = os.path.join(root_fdir, 'scripts')
utilities_fdir = os.path.join(scripts_fdir, 'utilities')
keras_fdir = os.path.join(scripts_fdir, 'keras')

for fdir in [root_fdir, data_fdir, scripts_fdir, utilities_fdir, keras_fdir]:
    sys.path.append(fdir)