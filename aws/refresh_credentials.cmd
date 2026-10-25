:: user settings
set user_dir=E:\GitHub\CatClassifier\.creds
set ubuntu_dir=/home/config
set session_fname=session_token.json

:: generate session token
call aws sts get-session-token > %user_dir%\%session_fname%