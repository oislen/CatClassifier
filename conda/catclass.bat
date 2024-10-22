:: create and activate new environment
call conda deactivate
call conda env remove --name catclass
call conda create --name catclass python=3.12 --yes
call conda activate catclass

:: update conda version
call conda update -n base conda --yes

:: install relevant libraries
call pip install -v -r ..\requirements.txt