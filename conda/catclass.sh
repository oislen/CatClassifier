# create and activate new environment
conda env remove --name catclass
conda create --name catclass python --yes
conda activate catclass

# update conda version
conda update -n base conda --yes

# install required webscraping packages
pip install Pillow
pip install playwright
conda install -c conda-forge requests --yes
conda install -c conda-forge beautifulsoup4 --yes
conda install -c conda-forge scrapy --yes

# install required data processing packages
conda install -c conda-forge numpy --yes
conda install -c conda-forge pandas --yes
pip install opencv-python

# install required model 
pip install scikit-learn
pip install tensorflow-cpu
pip install keras

# install required visualisation packages
conda install -c conda-forge matplotlib --yes
conda install -c conda-forge seaborn --yes

# run playwright install
playwright install

# deactivate new environment
conda deactivate