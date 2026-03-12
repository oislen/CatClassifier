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
    return test_df.to_dict(orient='records')[0]

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
    # check if the post request has the file part
    if 'image' not in request.files:
        return jsonify({"error": "No 'image' part in the request"}), 400
    # get the file object from the request
    file = request.files['image']
    # check if a file was actually selected for upload
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    # 4. If everything is fine, process the image
    if file:
        # You can save the file locally if needed:
        file.save(os.path.join('/path/to/save', file.filename))

        # Pass the file object itself to the (mock) classification function.
        # Note: 'file' is a Werkzeug FileStorage object. Your actual pipeline
        # might require the byte data (file.read()) or a PIL image object.
        # Ensure you adapt 'classify_image' to what your model needs.
        classification_result = classify_image(file)

        # 5. Return the classification result as a JSON response
        return jsonify(classification_result), 200
    else:
        # Handle cases where the file part might be present but invalid
        return jsonify({"error": "Invalid file upload"}), 400

if __name__ == '__main__':
    # run the flask application
    app.run(debug=True)