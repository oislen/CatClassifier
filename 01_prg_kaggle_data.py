# load relevant libraries
import os
import sys

# load custom modules
import cons
from download_comp_data import download_comp_data
     
# download competition data
download_comp_data(comp_name = cons.comp_name,
                   data_dir = cons.data_fdir,
                   download_data = cons.download_data, 
                   unzip_data = cons.unzip_data, 
                   del_zip = cons.del_zip
                   )
