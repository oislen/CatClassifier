import argparse

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
    # create an output dictionary to hold the results
    input_params_dict = {}
    # extract input arguments
    args = parser.parse_args()
    # map input arguments into output dictionary
    input_params_dict["run_model_training"] = args.run_model_training
    input_params_dict["run_testset_prediction"] = args.run_testset_prediction
    return input_params_dict
