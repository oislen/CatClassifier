:: set EC2 login info
SET EC2_USER=ec2-user
SET EC2_PEM_FPATH="C:\Users\oisin\.aws\kaggle.pem"
SET EC2_CREDS_FDIR=E:\GitHub\Cat-Classifier\.creds
SET EC2_SETUP_FPATH=E:\GitHub\Cat-Classifier\aws\linux_docker_setup.sh
SET EC2_EXTRACT_FPATH=E:\GitHub\Cat-Classifier\aws\docker_extract_data.sh
SET EC2_DNS_FPATH=%EC2_CREDS_FDIR%\ec2_dns

:: enable delyaed expansion to allow parsing of the %EC2_DNS_FPATH% file
SETLOCAL ENABLEDELAYEDEXPANSION
:: loop over %EC2_DNS_FPATH% lines and assign to variable
FOR /F "tokens=* USEBACKQ" %%F IN (`type %EC2_DNS_FPATH%`) DO (
    SET EC2_DNS!=%%F
)
:: execute extract docker data from windows
call ssh -i %EC2_PEM_FPATH% -t %EC2_USER%@%EC2_DNS% "bash ~/docker_extract_data.sh"
:: scp data files to local windows
call scp -i %EC2_PEM_FPATH% %EC2_SETUP_FPATH% %EC2_USER%@%EC2_DNS%:~/keras_model.h5 E:\GitHub\Cat-Classifier\data
call scp -i %EC2_PEM_FPATH% %EC2_SETUP_FPATH% %EC2_USER%@%EC2_DNS%:~/model_fit.pickle E:\GitHub\Cat-Classifier\data
call scp -i %EC2_PEM_FPATH% %EC2_SETUP_FPATH% %EC2_USER%@%EC2_DNS%:~/submission.csv E:\GitHub\Cat-Classifier\data
call scp -i %EC2_PEM_FPATH% -r %EC2_SETUP_FPATH% %EC2_USER%@%EC2_DNS%:~/checkpoints E:\GitHub\Cat-Classifier\data
:: scp report files to local windows
call scp -i %EC2_PEM_FPATH% %EC2_SETUP_FPATH% %EC2_USER%@%EC2_DNS%:~/generator_plot.jpg E:\GitHub\Cat-Classifier\report
call scp -i %EC2_PEM_FPATH% %EC2_SETUP_FPATH% %EC2_USER%@%EC2_DNS%:~/model_accuracy.png E:\GitHub\Cat-Classifier\report
call scp -i %EC2_PEM_FPATH% %EC2_SETUP_FPATH% %EC2_USER%@%EC2_DNS%:~/model_loss.png E:\GitHub\Cat-Classifier\report
call scp -i %EC2_PEM_FPATH% %EC2_SETUP_FPATH% %EC2_USER%@%EC2_DNS%:~/pred_images.jpg E:\GitHub\Cat-Classifier\report
call scp -i %EC2_PEM_FPATH% %EC2_SETUP_FPATH% %EC2_USER%@%EC2_DNS%:~/random_image.jpg E:\GitHub\Cat-Classifier\report
ENDLOCAL