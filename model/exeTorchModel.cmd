call SET PARAM_CHECK_GPU=True
:: call uv run python prg_torch_model.py --run_model_training --run_testset_prediction --model_id AlexNet8_pretrained
call uv run python prg_torch_model.py --run_model_training --run_testset_prediction --model_id VGG16_pretrained
:: call uv run python prg_torch_model.py --run_model_training --run_testset_prediction --model_id ResNet50_pretrained