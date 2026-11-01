export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export PARAM_CHECK_GPU=True
#uv run python prg_torch_model.py --run_model_training --run_testset_prediction --model_id alexnet_8_pretrained
uv run python prg_torch_model.py --run_model_training --run_testset_prediction --model_id vgg_16_pretrained
#uv run python prg_torch_model.py --run_model_training --run_testset_prediction --model_id resnet_50_pretrained