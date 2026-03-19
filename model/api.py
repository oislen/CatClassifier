import os
import logging
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from torch.utils.data import DataLoader
import datetime
import torch

import cons
from prg_torch_model import torch_transforms, model_dict
from model.torch.CustomDataset import CustomDataset
from model.utilities.TimeIt import TimeIt
from model.arch.load_image_v2 import TorchLoadImages

# set up logging
lgr = logging.getLogger()
lgr.setLevel(logging.INFO)
timeLogger = TimeIt()

app = Flask(__name__)

def classify_image(image_filepath:str, model_id:str, device_type:str) -> dict:
    """
    Classify an image using the pre-trained model.
    
    Parameters
    ----------
    image_filepath : str
        The file path of the image to classify.
    model_id : str
        The identifier of the model to use for classification.
    device_type : str
        The device type to run the model on (e.g., 'cpu' or 'cuda').
    
    Returns
    -------
    dict
        A dictionary containing the classification result and related information.
    """
    
    # set model architecture
    logging.info(f"model_id: {model_id}")
    logging.info(f"device_type: {device_type}")
    device = torch.device(device_type)
    model = model_dict[model_id].to(device)
    
    logging.info("Load fitted torch model from disk...")
    # load model
    model.load(input_fpath=cons.torch_model_pt_fpath.format(model_id=model_id))
    timeLogger.logTime(parentKey="ModelSerialisation", subKey="Load")
    
    logging.info("Generate test dataset...")
    # create torch load images object
    torchLoadImages = TorchLoadImages(torch_transforms=torch_transforms, n_workers=None)
    test_df = pd.DataFrame.from_records(torchLoadImages.loadImages(filepaths=[image_filepath]))
    logging.info(f"test_df.shape: {test_df.shape}")
    timeLogger.logTime(parentKey="DataPrep", subKey="TrainDataLoad")
    
    logging.info("Create test dataloader...")
    # set train data loader
    test_dataset = CustomDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=cons.batch_size, shuffle=False, num_workers=cons.num_workers, pin_memory=True, collate_fn=CustomDataset.collate_fn)
    timeLogger.logTime(parentKey="TestSet", subKey="DataLoader")
    
    logging.info("Generate test set predictions...")
    # make test set predictions
    predict = model.predict(test_loader, device)
    category = np.argmax(predict, axis=-1)
    test_df["category_id"] = category
    test_df["category_name"] = test_df["category_id"].replace(cons.category_mapper)
    test_df["probability"] = test_df["category_id"].apply(lambda x: predict[0][x])
    test_df = test_df.reset_index()
    # flush data from memory
    del test_dataset
    del test_loader
    timeLogger.logTime(parentKey="TestSet", subKey="ModelPredictions")
    # create api response
    sub_cols = ["index", "filepaths", "filenames", "category_id", "category_name", "probability", "ndims", "torch_transform_error"]
    response = test_df[sub_cols].to_dict(orient="records")[0]
    # add model id to repose
    response["model_id"] = model_id
    # append date time to response
    response["datetime"] = str(datetime.datetime.now())
    return response

@app.route("/catclassifier", methods=["POST"])
def endpoint():
    """
    API endpoint that accepts a POST request with an image file.
    Runs the image through the mock classification pipeline and returns the result.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    JSON response containing the classification result or an error message.
    """
    logging.info("Received POST request at /catclassifier endpoint")
    # check if the post request has the file part
    if "image" not in request.files:
        return jsonify({"error": "No 'image' part in the request"}), 400
    logging.info("Image file found in the request")
    # get the file object and target model from the request
    file = request.files["image"]
    model_id = request.form.get("model_id", list(model_dict.keys())[1])
    device_type = request.form.get("device_type", cons.device_type)
    if model_id not in model_dict:
        return jsonify({"error": f"Invalid model_id. Available model_ids: {list(model_dict.keys())}"}), 400
    logging.info(f"Received file: {file.filename}")
    # check if a file was actually selected for upload
    if file.filename == "":
        return jsonify({"error": "No file selected for uploading"}), 400
    # process the uploaded file and classify the image
    if file:
        logging.info("Processing the uploaded image")
        api_filepath = os.path.join(cons.api_fdir, file.filename)
        # create api file directory if not exists
        if not os.path.exists(cons.api_fdir):
            os.makedirs(os.path.dirname(api_filepath))
        # save the file uploaded to the api directory
        file.save(api_filepath)
        logging.info("Classifying the image using the model")
        # run the image through the classification pipeline
        classification_result = classify_image(image_filepath=api_filepath, model_id=model_id, device_type=device_type)
        # create response
        return jsonify(classification_result), 200
    else:
        # set default response
        return jsonify({"error": "Invalid file upload"}), 400

if __name__ == "__main__":
    # run the flask application
    app.run(debug=True)