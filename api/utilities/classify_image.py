import datetime
import logging

import mlflow
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import model.cons as model_cons
from model.arch.load_image_v2 import TorchLoadImages
from model.torch_models import CustomDataset
from model.train_torch_model import model_dict, torch_transforms
from model.utilities import TimeIt


def classify_image(
    image_filepath: str,
    model_id: str,
    device_type: str,
    original_filename: str = "",
    time_logger: TimeIt = TimeIt(),
    model_versions: dict = None,
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
    time_logger : TimeIt
        An instance of the TimeIt class for logging time taken for various steps.
    model_versions : dict
        A dictionary mapping model IDs to their respective version strings. If None, defaults to an empty

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
                    "registered_model_name": model_cons.mlflow_registered_model_names.get(
                        model_id, ""
                    ),
                    "model_version": model_versions.get(model_id, "unknown"),
                }
            )
            model.load(
                input_fpath=model_cons.torch_model_pt_fpath.format(model_id=model_id)
            )
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
            span.set_inputs({"batch_size": model_cons.batch_size})
            test_dataset = CustomDataset(test_df)
            test_loader = DataLoader(
                test_dataset,
                batch_size=model_cons.batch_size,
                shuffle=False,
                num_workers=0,
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
                model_cons.category_mapper
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
