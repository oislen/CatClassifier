import argparse
import cons

def commandline_interface():
    """A commandline interface for parsing input parameters with

    Parameters
    ----------

    Returns
    -------
    dict
        A dictionary of key, value pairs where the values are parsed input parameters
    """
    # define argument parser object
    parser = argparse.ArgumentParser(description="Execute Webscrapers.")
    # add input arguments
    parser.add_argument("--run_model_training", action=argparse.BooleanOptionalAction, dest="run_model_training", type=bool, default=False, help="Boolean, whether to run the model training pipeline, default is False",)
    parser.add_argument("--run_testset_prediction", action=argparse.BooleanOptionalAction, dest="run_testset_prediction", type=bool, default=False, help="Boolean, whether to run predictions on the test set, default is False",)
    parser.add_argument("--model_id", dest="model_id", type=str, default="VGG16_pretrained", choices=['AlexNet8_pretrained', 'VGG16_pretrained', 'ResNet50_pretrained'], help="String, id of the model architecture to use, default is VGG16_pretrained",)
    parser.add_argument("--device", dest="device", type=str, default=cons.device, choices=['cpu', 'cuda'], help="String, device to run the model on, default is cuda if available otherwise cpu",)
    # create an output dictionary to hold the results
    input_params_dict = {}
    # extract input arguments
    args = parser.parse_args()
    # map input arguments into output dictionary
    input_params_dict["run_model_training"] = args.run_model_training
    input_params_dict["run_testset_prediction"] = args.run_testset_prediction
    input_params_dict["model_id"] = args.model_id
    input_params_dict["device"] = args.device
    return input_params_dict
