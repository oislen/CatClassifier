import logging
import os
import threading

import mlflow
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

import model.cons as cons
from api.utilities import classify_image
from model.train_torch_model import model_dict, train_torch_model
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


@app.route("/catclassifier/image", methods=["POST"])
def image_endpoint():
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
    logging.info("Received POST request at /catclassifier/image endpoint")
    # early exits before the span — no image data yet
    if "image" not in request.files:
        return jsonify({"error": "No 'image' part in the request"}), 400
    logging.info("Image file found in the request")

    # extract parameters from the request
    image_file = request.files.get("image", None)
    model_id = request.form.get("model_id", list(model_dict.keys())[1])
    device_type = request.form.get("device_type", cons.device_type)
    safe_filename = (
        secure_filename(image_file.filename) if image_file is not None else ""
    )

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
                "registered_model_name": cons.mlflow_registered_model_names.get(
                    model_id, ""
                ),
                "model_version": _MODEL_VERSIONS.get(model_id, "unknown"),
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
                time_logger=time_logger,
                model_versions=_MODEL_VERSIONS,
            )
        except Exception as exc:
            root_span.set_outputs({"status_code": 500, "error": str(exc)})
            return (
                jsonify({"error": "Internal classification error", "detail": str(exc)}),
                500,
            )

        # include the full result dict alongside status code for rich trace output
        root_span.set_outputs({"status_code": 200, **classification_result})

    return jsonify(classification_result), 200


@app.route("/catclassifier/train", methods=["POST"])
def train_endpoint():
    """
    API endpoint that accepts a POST request to initiate model training or test set prediction.

    Parameters
    ----------
    None

    Returns
    -------
    JSON response indicating the initiation of model training or test set prediction, or an error message.
    """
    logging.info("Received POST request at /catclassifier/train endpoint")

    # extract parameters from the request
    model_id = request.form.get("model_id", list(model_dict.keys())[1])
    device_type = request.form.get("device_type", cons.device_type)
    run_model_training = (
        request.form.get("run_model_training", "False").lower() == "true"
    )
    run_testset_prediction = (
        request.form.get("run_testset_prediction", "False").lower() == "true"
    )

    # open root span — all remaining logic (including model_id validation) runs inside
    with mlflow.start_span(name="api_request", span_type="CHAIN") as root_span:
        root_span.set_inputs(
            {
                "model_id": model_id,
                "device_type": device_type,
                "git_commit": _GIT_COMMIT,
                "request_id": request.headers.get("X-Request-ID", "none"),
                "run_model_training": run_model_training,
                "run_testset_prediction": run_testset_prediction,
                "registered_model_name": cons.mlflow_registered_model_names.get(
                    model_id, ""
                ),
                "model_version": _MODEL_VERSIONS.get(model_id, "unknown"),
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

        # set up input parameters for model training or test set prediction
        input_params_dict = {
            "model_id": model_id,
            "device_type": device_type,
            "run_model_training": run_model_training,
            "run_testset_prediction": run_testset_prediction,
        }
        # Run model training or test set prediction if requested
        try:
            logging.info(
                "Starting model training or test set prediction in a separate thread"
            )
            # Start a new thread to run the training or prediction function
            train_thread = threading.Thread(
                target=train_torch_model, args=(input_params_dict,)
            )
            train_thread.start()
            classification_result = {
                "message": "Model training or test set prediction initiated successfully."
            }
            root_span.set_outputs({"status_code": 200, **classification_result})
        except Exception as exc:
            logging.exception(f"Model training/prediction failed: {exc}")
            root_span.set_outputs({"status_code": 500, "error": str(exc)})
            return (
                jsonify(
                    {
                        "error": "Internal training/prediction initiation error",
                        "detail": str(exc),
                    }
                ),
                500,
            )

    return jsonify(classification_result), 200


if __name__ == "__main__":
    # Determine debug mode from environment variable, defaulting to False
    debug_mode = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    # run the flask application
    app.run(debug=debug_mode)
