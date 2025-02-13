:: create and activate new environment
call conda deactivate
call conda env remove --name catclass
call conda create --name catclass python=3.12 --yes
call conda activate catclass

:: update conda version
call conda update -n base conda --yes

:: install relevant libraries
call pip install -v -r ..\requirements.txt
call pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121