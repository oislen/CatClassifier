import logging
import os
import platform

import mlflow
import numpy as np
import pandas as pd

# set huggingface hub directory
huggingface_hub_dir = "E:\\huggingface"
if (platform.system() == "Windows") and (os.path.exists(huggingface_hub_dir)):
    os.environ["TORCH_HOME"] = huggingface_hub_dir
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# load custom scripts
import cons
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from model.arch.load_image_v2 import TorchLoadImages
from model.torch_models import (
    CustomDataset,
    EarlyStopper,
    alexnet_8_pretrained,
    export_onnx,
    resnet_50_pretrained,
    vgg_16_pretrained,
)
from model.utilities import (
    TimeIt,
    commandline_interface,
    compute_eval_metrics,
    plot_generator,
    plot_image,
    plot_model_fit,
    plot_preds,
)

# initialise model dict
model_dict = {
    "alexnet_8_pretrained": alexnet_8_pretrained(num_classes=2),
    "vgg_16_pretrained": vgg_16_pretrained(num_classes=2),
    "resnet_50_pretrained": resnet_50_pretrained(num_classes=2),
}

random_state = 42

torch_transforms = transforms.Compose(
    [
        transforms.Resize(
            size=[cons.IMAGE_WIDTH, cons.IMAGE_HEIGHT]
        ),  # resize the input image to a uniform size
        # ,transforms.RandomRotation(30)
        # ,transforms.RandomHorizontalFlip(p=0.05)
        # ,transforms.RandomPerspective(distortion_scale=0.05, p=0.05)
        transforms.ToTensor(),  # convert PIL Image or numpy.ndarray to tensor and normalize to somewhere between [0,1]
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # standardized processing
    ]
)

if __name__ == "__main__":
    # set up logging
    lgr = logging.getLogger()
    lgr.setLevel(logging.INFO)
    time_logger = TimeIt()

    logging.info("Parsing command line arguments...")
    # handle input parameters
    input_params_dict = commandline_interface()
    logging.info(input_params_dict)
    time_logger.log_time(parent_key="Initialisation", sub_key="CommandlineArguments")

    # set model architecture
    model_id = input_params_dict["model_id"]
    device_type = input_params_dict["device_type"]
    logging.info(f"model_id: {model_id}")
    logging.info(f"device_type: {device_type}")
    device = torch.device(input_params_dict["device_type"])
    model = model_dict[model_id].to(device)

    # configure mlflow tracking
    mlflow.set_tracking_uri(cons.mlflow_tracking_uri)
    mlflow.set_experiment(cons.mlflow_experiment_name)

    # read run_id at call time to support run resumption across separate invocations
    mlflow_run_id = os.environ.get("MLFLOW_RUN_ID", None)
    with mlflow.start_run(run_id=mlflow_run_id, run_name=model_id) as run:
        logging.info(f"MLflow run_id: {run.info.run_id}")

        if input_params_dict["run_model_training"]:
            logging.info("Generating dataframe of images...")
            # load and shuffle the image file paths
            np.random.seed(random_state)
            image_filepaths = np.array(
                [os.path.join(cons.train_fdir, x) for x in os.listdir(cons.train_fdir)]
            )
            np.random.shuffle(image_filepaths)
            # create torch load images object
            sample_size = 20000
            torch_load_images = TorchLoadImages(
                torch_transforms=torch_transforms, n_workers=None
            )
            df = pd.DataFrame.from_records(
                torch_load_images.load_images(image_filepaths[0:sample_size])
            )
            # only consider images with 3 dimensions
            df = df.loc[df["ndims"] == 3, :]
            # flush data from memory
            del image_filepaths
            logging.info(f"df.shape: {df.shape}")
            time_logger.log_time(parent_key="DataPrep", sub_key="TrainDataLoad")

            logging.info("Plot sample image...")
            # random image plot
            plot_image(
                df["images"].values[1],
                output_fpath=cons.torch_random_image_fpath,
                show_plot=False,
            )
            time_logger.log_time(parent_key="Plots", sub_key="SampleImage")

            logging.info("Plot example data loader images...")
            # data generator example
            plot_generator(
                generator=df["image_tensors"].values[:16].tolist(),
                mode="torch",
                output_fpath=cons.torch_generator_plot_fpath,
                show_plot=False,
            )
            time_logger.log_time(parent_key="Plots", sub_key="DataLoader")

            logging.info("Split into training, validation and test dataset...")
            # prepare data
            validate_df = df.sample(frac=0.05, random_state=random_state, replace=False)
            train_df = df[~df.index.isin(validate_df.index)]
            train_df = train_df.reset_index(drop=True)
            validate_df = validate_df.reset_index(drop=True)
            logging.info(f"train_df.shape: {train_df.shape}")
            logging.info(f"validate_df.shape: {validate_df.shape}")
            time_logger.log_time(parent_key="DataPrep", sub_key="TrainValidationSplit")

            logging.info("Creating training and validation data loaders...")
            # set data constants
            total_train = train_df.shape[0]
            total_validate = validate_df.shape[0]
            # set train data loader
            train_dataset = CustomDataset(train_df)
            train_loader = DataLoader(
                train_dataset,
                batch_size=cons.batch_size,
                shuffle=True,
                num_workers=cons.num_workers,
                pin_memory=True,
                collate_fn=CustomDataset.collate_fn,
            )
            # set validation data loader
            validation_dataset = CustomDataset(validate_df)
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=cons.batch_size,
                shuffle=True,
                num_workers=cons.num_workers,
                pin_memory=True,
                collate_fn=CustomDataset.collate_fn,
            )
            # flush data from memory
            del df
            del train_df
            del train_dataset
            del validate_df
            del validation_dataset
            time_logger.log_time(
                parent_key="DataPrep", sub_key="TrainValidationDataLoaders"
            )

            logging.info("Initiate torch model...")
            # initiate cnn architecture
            if device == "cuda":
                model = nn.DataParallel(model)
            model = model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=cons.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=10,
                threshold=0.0001,
                threshold_mode="abs",
            )
            early_stopper = EarlyStopper(patience=3, min_delta=0.3)
            time_logger.log_time(parent_key="Modelling", sub_key="InitiateTorchModel")

            logging.info("Fit torch model...")
            # hyper-parameters
            num_epochs = cons.min_epochs if cons.FAST_RUN else cons.max_epochs
            # log hyperparameters on fresh runs only (params are immutable in MLflow)
            if mlflow_run_id is None:
                mlflow.log_params({
                    "model_id": model_id,
                    "num_epochs": num_epochs,
                    "batch_size": cons.batch_size,
                    "learning_rate": cons.learning_rate,
                    "image_width": cons.IMAGE_WIDTH,
                    "image_height": cons.IMAGE_HEIGHT,
                    "random_state": random_state,
                    "device_type": device_type,
                })
            # fit torch model
            model.fit(
                device=device,
                criterion=criterion,
                optimizer=optimizer,
                train_dataloader=train_loader,
                num_epochs=num_epochs,
                scheduler=scheduler,
                valid_data_loader=validation_loader,
                early_stopper=early_stopper,
                checkpoints_dir=cons.checkpoints_fdir,
                load_epoch_checkpoint=None,
            )
            # flush data from memory
            del train_loader
            del validation_loader
            time_logger.log_time(parent_key="Modelling", sub_key="Fit")
            mlflow.log_metric("training_time_seconds", time_logger.cumulative_time)

            logging.info("Plot model fit results...")
            # plot model fits
            plot_model_fit(
                model_fit=model.model_fit,
                output_fdir=cons.torch_report_fdir,
                show_plot=False,
            )
            time_logger.log_time(parent_key="Plots", sub_key="ModelFit")

            logging.info("Save fitted torch model to disk...")
            # save model
            pt_fpath = cons.torch_model_pt_fpath.format(model_id=model.model_id)
            model.save(output_fpath=pt_fpath)
            time_logger.log_time(parent_key="ModelSerialisation", sub_key="Write")

            logging.info("Export fitted torch model to ONNX...")
            onnx_fpath = cons.torch_model_onnx_fpath.format(model_id=model.model_id)
            export_onnx(model, onnx_fpath, cons.IMAGE_WIDTH, cons.IMAGE_HEIGHT)
            time_logger.log_time(parent_key="ModelSerialisation", sub_key="OnnxExport")

            logging.info("Log model artefacts to MLflow...")
            mlflow.log_artifact(pt_fpath)
            mlflow.log_artifact(onnx_fpath)

        if input_params_dict["run_testset_prediction"]:
            logging.info("Load fitted torch model from disk...")
            # load model
            model.load(
                input_fpath=cons.torch_model_pt_fpath.format(model_id=model.model_id)
            )
            time_logger.log_time(parent_key="ModelSerialisation", sub_key="Load")

            logging.info("Generate test dataset...")
            # create torch load images object
            torch_load_images = TorchLoadImages(
                torch_transforms=torch_transforms, n_workers=None
            )
            test_df = pd.DataFrame.from_records(
                torch_load_images.load_images(
                    filepaths=[
                        os.path.join(cons.test_fdir, x) for x in os.listdir(cons.test_fdir)
                    ]
                )
            )
            logging.info(f"test_df.shape: {test_df.shape}")
            time_logger.log_time(parent_key="DataPrep", sub_key="TrainDataLoad")

            logging.info("Create test dataloader...")
            # set train data loader
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

            logging.info("Generate test set predictions...")
            # make test set predictions
            predict = model.predict(test_loader, device)
            test_df["category"] = np.argmax(predict, axis=-1)
            test_df["category"] = test_df["category"].replace(cons.category_mapper)
            # flush data from memory
            del test_dataset
            del test_loader
            time_logger.log_time(parent_key="TestSet", sub_key="ModelPredictions")

            logging.info("Plot example test set predictions...")
            # plot random sample predictions
            plot_preds(
                data=test_df, output_fpath=cons.torch_pred_images_fpath, show_plot=False
            )
            time_logger.log_time(parent_key="Plots", sub_key="TestSetPredictions")

            logging.info("Generate a sample submission file for kaggle...")
            # make submission
            submission_df = test_df.copy()
            submission_df["id"] = submission_df["filenames"].str.split(".").str[0]
            submission_df["label"] = submission_df["category"].replace(cons.category_mapper)
            submission_df.to_csv(cons.submission_csv_fpath, index=False)
            time_logger.log_time(parent_key="TestSet", sub_key="SubmissionFile")

            logging.info("Compute and log evaluation metrics to MLflow...")
            # derive ground-truth labels from filenames (cat.N.jpg -> cat, dog.N.jpg -> dog)
            test_df["true_label"] = test_df["filenames"].str.split(".").str[0]
            known_classes = set(cons.category_mapper.values())  # {"cat", "dog"}
            if set(test_df["true_label"].unique()).issubset(known_classes):
                eval_metrics = compute_eval_metrics(test_df["true_label"], test_df["category"])
                mlflow.log_metrics(eval_metrics)
                logging.info(f"Evaluation metrics: {eval_metrics}")
            else:
                logging.info(
                    "Skipping evaluation metrics: test filenames do not contain "
                    "class labels (Kaggle unlabelled test set detected)."
                )

            logging.info("Log evaluation artefacts to MLflow...")
            mlflow.log_artifact(cons.submission_csv_fpath)
            mlflow.log_artifact(cons.torch_pred_images_fpath)

            # delete dataframes from memory
            del test_df
            del submission_df
