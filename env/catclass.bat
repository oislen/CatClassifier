:: create and activate new environment
call conda env remove --name catclass
call conda create --name catclass python --yes
call conda activate catclass

:: update conda version
call conda update -n base conda --yes

:: conda install required webscraping packages
call pip install Pillow
call pip install playwright
call conda install -c conda-forge requests --yes
call conda install -c conda-forge beautifulsoup4 --yes
call conda install -c conda-forge scrapy --yes

:: conda install required data processing packages
call conda install -c conda-forge numpy --yes
call conda install -c conda-forge pandas --yes
call pip install opencv-python

:: conda install equired visualisation packages
call conda install -c conda-forge matplotlib --yes
call conda install -c conda-forge seaborn --yes

:: run playwright install
call playwright install

:: deactivate new environment
call conda deactivate