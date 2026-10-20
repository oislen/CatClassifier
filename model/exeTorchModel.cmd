call SET PARAM_CHECK_GPU=True
:: call uv run python prg_torch_model.py --run_model_training --run_testset_prediction --model_id alexnet_8_pretrained --device_type cuda
call uv run python prg_torch_model.py --run_model_training --run_testset_prediction --model_id vgg_16_pretrained --device_type cuda
:: call uv run python prg_torch_model.py --run_model_training --run_testset_prediction --model_id resnet_50_pretrained --device_type cuda