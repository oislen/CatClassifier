:: call rmdir /s /q ..\..\.venv
:: call powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
:: uv init CatClassifier
call uv add -r ..\..\requirements.txt --link-mode=copy
call uv add --index https://download.pytorch.org/whl/cu129 torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --link-mode=copy