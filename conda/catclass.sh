# create and activate new environment
conda deactivate
conda env remove --name catclass
conda create --name catclass python3 --yes
conda activate catclass

# update conda version
conda update -n base conda --yes

# install relevant libraries
pip install -v -r ..\requirements.txt