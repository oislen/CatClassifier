# create and activate new environment
conda deactivate
conda env remove --name catclass
conda create --name catclass python --yes
conda activate catclass

# update conda version
conda update -n base conda --yes

# install relevant libraries
pip install -r ..\requirements.txt

# run playwright install
playwright install

# install jupyterlab
conda install -c conda-forge jupyterlab --yes

# deactivate new environment
conda deactivate