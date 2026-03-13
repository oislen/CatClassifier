import os
import logging
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from torch.utils.data import DataLoader

import cons
from prg_torch_model import device, model, torch_transforms
from model.torch.CustomDataset import CustomDataset
from model.utilities.TimeIt import TimeIt
from model.arch.load_image_v2 import TorchLoadImages

# set up logging
lgr = logging.getLogger()
lgr.setLevel(logging.INFO)
timeLogger = TimeIt()

app = Flask(__name__)

def classify_image(image_filepath):
    """
    """
    logging.info("Load fitted torch model from disk...")
    # load model
    model.load(input_fpath=cons.torch_model_pt_fpath)
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
    test_df['category'] = np.argmax(predict, axis=-1)
    test_df["category"] = test_df["category"].replace(cons.category_mapper)
    # flush data from memory
    del test_dataset
    del test_loader
    timeLogger.logTime(parentKey="TestSet", subKey="ModelPredictions")
    # create api response
    sub_cols = ['filepaths', 'filenames', 'category', 'ndims', 'torch_transform_error']
    response = test_df[sub_cols].to_dict(orient='records')[0]
    return response

@app.route('/catclassifier', methods=['POST'])
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
    if 'image' not in request.files:
        response = jsonify({"error": "No 'image' part in the request"}), 400
    else:
        logging.info("Image file found in the request")
        # get the file object from the request
        file = request.files['image']
        logging.info(f"Received file: {file.filename}")
        # check if a file was actually selected for upload
        if file.filename == '':
            response = jsonify({"error": "No file selected for uploading"}), 400
        else:
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
                classification_result = classify_image(image_filepath=api_filepath)
                # create response
                response = jsonify(classification_result), 200
            else:
                # set default response
                response = jsonify({"error": "Invalid file upload"}), 400
    # return response
    return response

if __name__ == '__main__':
    # run the flask application
    app.run(debug=True)