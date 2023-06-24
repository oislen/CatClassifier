:: create and activate new environment
call conda deactivate
call conda env remove --name catclass
call conda create --name catclass python --yes
call conda activate catclass

:: update conda version
call conda update -n base conda --yes

:: install relevant libraries
call pip install -r ..\requirements.txt

:: run playwright install
call playwright install

:: deactivate new environment
call conda deactivate