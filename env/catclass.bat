:: create and activate new environment
call conda env remove --name catclass
call conda create --name catclass python --yes
call conda activate catclass

:: update conda version
call conda update -n base conda --yes

:: pip install required packages
call pip install playwright

:: conda install required webscraping packages
call conda install -c conda-forge requests --yes
call conda install -c conda-forge beautifulsoup4 --yes
call conda install -c conda-forge scrapy --yes

:: run playwright install
call playwright install

:: deactivate new environment
call conda deactivate