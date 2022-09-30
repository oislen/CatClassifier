:: create and activate new environment
call conda env remove --name catclass
call conda create --name catclass python --yes
call conda activate catclass

:: update conda version
call conda update -n base conda --yes

:: install required webscraping packages
call pip install Pillow
call pip install playwright
call conda install -c conda-forge requests --yes
call conda install -c conda-forge beautifulsoup4 --yes
call conda install -c conda-forge scrapy --yes

:: install required data processing packages
call conda install -c conda-forge numpy --yes
call conda install -c conda-forge pandas --yes
call pip install opencv-python

:: install required model 
call pip install scikit-learn
call pip install tensorflow-cpu
call pip install keras

:: install required visualisation packages
call conda install -c conda-forge matplotlib --yes
call conda install -c conda-forge seaborn --yes

:: run playwright install
call playwright install

:: install jupyterlab
call conda install -c conda-forge jupyterlab --yes

:: deactivate new environment
call conda deactivate