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
    parser.add_argument("--run_download_models", action=argparse.BooleanOptionalAction, dest="run_download_models", type=bool, default=False, help="Boolean, whether to run the download master Kaggle models, default is False",)
    parser.add_argument("--run_download_comp_data", action=argparse.BooleanOptionalAction, dest="run_download_comp_data", type=bool, default=False, help="Boolean, whether to run the download Kaggle competition data, default is False",)
    parser.add_argument("--run_webscraper", action=argparse.BooleanOptionalAction, dest="run_webscraper", type=bool, default=False, help="Boolean, whether to run the image webscraper, default is False",)
    # create an output dictionary to hold the results
    input_params_dict = {}
    # extract input arguments
    args = parser.parse_args()
    # map input arguments into output dictionary
    input_params_dict["run_download_models"] = args.run_download_models
    input_params_dict["run_download_comp_data"] = args.run_download_comp_data
    input_params_dict["run_webscraper"] = args.run_webscraper
    return input_params_dict
