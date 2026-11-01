import datetime
import logging
import os

import cons
import mlflow
import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, request
from prg_torch_model import model_dict, torch_transforms
from torch.utils.data import DataLoader
from werkzeug.utils import secure_filename

from model.arch.load_image_v2 import TorchLoadImages
from model.torch_models import CustomDataset
from model.utilities import TimeIt, get_git_commit_sha

# set up logging
lgr = logging.getLogger()
lgr.setLevel(logging.INFO)
time_logger = TimeIt()

# ---------------------------------------------------------------------------
# MLflow startup configuration — runs once when the Flask process starts
# ---------------------------------------------------------------------------

# configure MLflow — tracing only, no run context (inference requests are not runs)
mlflow.set_tracking_uri(cons.mlflow_tracking_uri)
mlflow.set_experiment(cons.mlflow_experiment_name)

# cache git SHA to avoid a subprocess call on every request
_GIT_COMMIT = get_git_commit_sha()

# MLflow client for Model Registry lookups
_client = mlflow.tracking.MlflowClient()


def _get_latest_model_version(model_id: str) -> str:
    """Return the latest version string for the registered model, or 'unknown'."""
    registered_name = cons.mlflow_registered_model_names.get(model_id, "")
    if not registered_name:
        return "unknown"
    try:
        versions = _client.get_registered_model(registered_name).latest_versions
        return versions[0].version if versions else "unknown"
    except Exception:
        return "unknown"


# pre-cache all registered model versions at startup — avoids per-request Registry queries
_MODEL_VERSIONS = {
    mid: _get_latest_model_version(mid) for mid in cons.mlflow_registered_model_names
}

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)


def classify_image(
    image_filepath: str,
    model_id: str,
    device_type: str,
    original_filename: str = "",
) -> dict:
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
    original_filename : str
        The original uploaded filename (for tracing/debugging). Default is "".

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

    try:
        # --- model load span ---
        with mlflow.start_span(name="model_load", span_type="TASK") as span:
            logging.info("Load fitted torch model from disk...")
            span.set_inputs({"model_id": model_id, "device_type": device_type})
            span.set_attributes(
                {
                    "registered_model_name": cons.mlflow_registered_model_names.get(
                        model_id, ""
                    ),
                    "model_version": _MODEL_VERSIONS.get(model_id, "unknown"),
                }
            )
            model.load(input_fpath=cons.torch_model_pt_fpath.format(model_id=model_id))
            time_logger.log_time(parent_key="ModelSerialisation", sub_key="Load")
            span.set_outputs({"loaded": True})

        # --- image loading span ---
        with mlflow.start_span(name="image_loading", span_type="TASK") as span:
            logging.info("Generate test dataset...")
            span.set_inputs(
                {
                    "image_filepath": image_filepath,
                    "original_filename": original_filename,
                }
            )
            torch_load_images = TorchLoadImages(
                torch_transforms=torch_transforms, n_workers=None
            )
            test_df = pd.DataFrame.from_records(
                torch_load_images.load_images(filepaths=[image_filepath])
            )
            logging.info(f"test_df.shape: {test_df.shape}")
            time_logger.log_time(parent_key="DataPrep", sub_key="TrainDataLoad")
            span.set_outputs(
                {"ndims": int(test_df["ndims"].values[0]) if len(test_df) > 0 else -1}
            )

        # --- dataloader span ---
        with mlflow.start_span(name="dataloader_creation", span_type="TASK") as span:
            logging.info("Create test dataloader...")
            span.set_inputs({"batch_size": cons.batch_size})
            test_dataset = CustomDataset(test_df)
            test_loader = DataLoader(
                test_dataset,
                batch_size=cons.batch_size,
                shuffle=False,
                num_workers=cons.num_workers,
                pin_memory=True,
                collate_fn=CustomDataset.collate_fn,
            )
            time_logger.log_time(parent_key="TestSet", sub_key="DataLoader")
            span.set_outputs({"num_batches": len(test_loader)})

        # --- prediction span ---
        with mlflow.start_span(name="prediction", span_type="TASK") as span:
            logging.info("Generate test set predictions...")
            span.set_inputs({"model_id": model_id})
            predict = model.predict(test_loader, device)
            category = np.argmax(predict, axis=-1)
            test_df["category_id"] = category
            test_df["category_name"] = test_df["category_id"].replace(
                cons.category_mapper
            )
            test_df["probability"] = test_df["category_id"].apply(
                lambda x: predict[0][x]
            )
            test_df = test_df.reset_index()
            # flush data from memory
            del test_dataset
            del test_loader
            time_logger.log_time(parent_key="TestSet", sub_key="ModelPredictions")
            span.set_outputs(
                {
                    "category_id": int(category[0]),
                    "category_name": str(test_df["category_name"].values[0]),
                    "probability": float(test_df["probability"].values[0]),
                }
            )

    except Exception as exc:
        logging.exception(f"classify_image failed: {exc}")
        raise  # re-raise so the root span in endpoint() marks itself ERROR and returns 500

    # create api response
    sub_cols = [
        "index",
        "filepaths",
        "filenames",
        "category_id",
        "category_name",
        "probability",
        "ndims",
        "torch_transform_error",
    ]
    response = test_df[sub_cols].to_dict(orient="records")[0]
    # add model id to response
    response["model_id"] = model_id
    # append date time to response
    response["datetime"] = str(datetime.datetime.now())
    return response


@app.route("/catclassifier", methods=["POST"])
def endpoint():
    """
    API endpoint that accepts a POST request with an image file.
    Runs the image through the classification pipeline and returns the result.

    Parameters
    ----------
    None

    Returns
    -------
    JSON response containing the classification result or an error message.
    """
    logging.info("Received POST request at /catclassifier endpoint")
    # early exits before the span — no image data yet
    if "image" not in request.files:
        return jsonify({"error": "No 'image' part in the request"}), 400
    logging.info("Image file found in the request")

    image_file = request.files["image"]
    model_id = request.form.get("model_id", list(model_dict.keys())[1])
    device_type = request.form.get("device_type", cons.device_type)
    safe_filename = secure_filename(image_file.filename)

    if safe_filename == "":
        return jsonify({"error": "No file selected for uploading"}), 400

    # open root span — all remaining logic (including model_id validation) runs inside
    with mlflow.start_span(name="api_request", span_type="CHAIN") as root_span:
        root_span.set_inputs(
            {
                "model_id": model_id,
                "device_type": device_type,
                "image_filename": safe_filename,
                "git_commit": _GIT_COMMIT,
                "request_id": request.headers.get("X-Request-ID", "none"),
            }
        )

        if model_id not in model_dict:
            root_span.set_outputs({"status_code": 400, "error": "invalid model_id"})
            return (
                jsonify(
                    {
                        "error": f"Invalid model_id. Available model_ids: {list(model_dict.keys())}"
                    }
                ),
                400,
            )

        logging.info("Processing the uploaded image")
        api_filepath = os.path.join(cons.api_fdir, safe_filename)
        # create api file directory if not exists
        if not os.path.exists(cons.api_fdir):
            os.makedirs(os.path.dirname(api_filepath))
        # save the file uploaded to the api directory
        image_file.save(api_filepath)
        logging.info("Classifying the image using the model")

        try:
            classification_result = classify_image(
                image_filepath=api_filepath,
                model_id=model_id,
                device_type=device_type,
                original_filename=safe_filename,
            )
        except Exception as exc:
            root_span.set_outputs({"status_code": 500, "error": str(exc)})
            return jsonify(
                {"error": "Internal classification error", "detail": str(exc)}
            ), 500

        # include the full result dict alongside status code for rich trace output
        root_span.set_outputs({"status_code": 200, **classification_result})

    return jsonify(classification_result), 200


if __name__ == "__main__":
    # Determine debug mode from environment variable, defaulting to False
    debug_mode = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    # run the flask application
    app.run(debug=debug_mode)
