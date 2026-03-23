export PARAM_CHECK_GPU=True
#uv run python prg_torch_model.py --run_model_training --run_testset_prediction --model_id AlexNet8_pretrained
uv run python prg_torch_model.py --run_model_training --run_testset_prediction --model_id VGG16_pretrained
#uv run python prg_torch_model.py --run_model_training --run_testset_prediction --model_id ResNet50_pretrained