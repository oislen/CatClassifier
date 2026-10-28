"""
Smoke tests for the MLflow integration in prg_torch_model.py.

These tests exercise the full training and prediction pipeline end-to-end
against a tiny synthetic dataset and a file-based MLflow tracking store
(no HTTP server required). They are intentionally excluded from the normal
unittest discovery path (model/unittests/) and should be run manually:

    uv run python -m unittest smoketests.test_smoke_mlflow

from the CatClassifier/model/ directory.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cons
import mlflow
from torch_models import EarlyStopper, ModelFit, export_onnx
from torch_models.fit import fit as fit_fn
from utilities.evaluate import compute_eval_metrics

# ---------------------------------------------------------------------------
# Minimal model shared across all smoke tests
# ---------------------------------------------------------------------------

IMAGE_WIDTH = 16
IMAGE_HEIGHT = 16
NUM_CLASSES = 2
BATCH_SIZE = 4
NUM_SAMPLES = 16


class _SmallNet(nn.Module):
    """Tiny conv-free net for fast CPU smoke testing."""

    def __init__(self):
        super().__init__()
        self.model_id = "small_net"
        self.fc = nn.Linear(3 * IMAGE_WIDTH * IMAGE_HEIGHT, NUM_CLASSES)
        self.model_fit = None

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

    def fit(self, device, criterion, optimizer, train_dataloader, num_epochs=1,
            scheduler=None, valid_data_loader=None, early_stopper=None,
            checkpoints_dir=None, load_epoch_checkpoint=None):
        self, self.model_fit = fit_fn(
            self, device, criterion, optimizer, train_dataloader, num_epochs,
            scheduler, valid_data_loader, early_stopper, checkpoints_dir,
            load_epoch_checkpoint,
        )

    def predict(self, dataloader, device):
        self.eval()
        preds = []
        with torch.no_grad():
            for images, _ in dataloader:
                preds.append(self.forward(images.to(device)).cpu().numpy())
        return np.concatenate(preds, axis=0)

    def save(self, output_fpath):
        torch.save(self.state_dict(), output_fpath)
        return f"Saved to {output_fpath}"

    def load(self, input_fpath, weights_only=False):
        self.load_state_dict(torch.load(input_fpath, weights_only=weights_only))
        self.eval()
        return f"Loaded from {input_fpath}"


def _make_dataloader():
    """Return a tiny DataLoader with random float tensors and binary labels."""
    images = torch.randn(NUM_SAMPLES, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    labels = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
    ds = TensorDataset(images, labels)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)


# ---------------------------------------------------------------------------
# Base class: sets up an isolated file-based MLflow tracking store
# ---------------------------------------------------------------------------


class _SmokeBase(unittest.TestCase):
    """Sets up and tears down an isolated MLflow tracking store per test."""

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.mkdtemp()
        cls._tracking_uri = f"file://{cls._tmpdir}"
        mlflow.set_tracking_uri(cls._tracking_uri)
        mlflow.set_experiment(cons.mlflow_experiment_name)
        cls._run_id = cls._run_pipeline()

    @classmethod
    def _run_pipeline(cls):
        """Execute a minimal training + prediction pipeline inside one MLflow run."""
        device = torch.device("cpu")
        model = _SmallNet().to(device)
        train_loader = _make_dataloader()
        valid_loader = _make_dataloader()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        with mlflow.start_run(run_name="smoke_test") as run:
            run_id = run.info.run_id
            mlflow.log_params({
                "model_id": model.model_id,
                "num_epochs": 1,
                "batch_size": BATCH_SIZE,
                "learning_rate": 0.01,
                "image_width": IMAGE_WIDTH,
                "image_height": IMAGE_HEIGHT,
                "random_state": 42,
                "device_type": "cpu",
            })

            # --- training ---
            model.fit(
                device=device,
                criterion=criterion,
                optimizer=optimizer,
                train_dataloader=train_loader,
                num_epochs=1,
                valid_data_loader=valid_loader,
            )

            # --- serialisation ---
            pt_path = os.path.join(cls._tmpdir, f"{model.model_id}.pt")
            onnx_path = os.path.join(cls._tmpdir, f"{model.model_id}.onnx")
            model.save(pt_path)
            export_onnx(model, onnx_path, IMAGE_WIDTH, IMAGE_HEIGHT)
            mlflow.log_artifact(pt_path)
            mlflow.log_artifact(onnx_path)

            # --- evaluation ---
            test_loader = _make_dataloader()
            predict = model.predict(test_loader, device)
            categories = np.argmax(predict, axis=-1)
            # synthesise string labels matching the cat/dog convention
            pred_labels = ["dog" if c == 1 else "cat" for c in categories]
            # derive true labels from positional parity (deterministic fixture)
            true_labels = ["dog" if i % 2 == 1 else "cat" for i in range(len(pred_labels))]
            eval_metrics = compute_eval_metrics(true_labels, pred_labels)
            mlflow.log_metrics(eval_metrics)

            # --- submission CSV artefact ---
            csv_path = os.path.join(cls._tmpdir, "submission.csv")
            import pandas as pd
            pd.DataFrame({"id": range(NUM_SAMPLES), "label": pred_labels}).to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path)

        return run_id

    def _get_run(self):
        return mlflow.get_run(self._run_id)

    def _list_artefacts(self):
        client = mlflow.tracking.MlflowClient()
        return [a.path for a in client.list_artifacts(self._run_id)]


# ---------------------------------------------------------------------------
# Tracking tests
# ---------------------------------------------------------------------------


class TestSmokeTracking(_SmokeBase):

    def test_smoke_mlflow_experiment_created(self):
        exp = mlflow.get_experiment_by_name(cons.mlflow_experiment_name)
        self.assertIsNotNone(exp)

    def test_smoke_mlflow_run_created(self):
        client = mlflow.tracking.MlflowClient()
        exp = mlflow.get_experiment_by_name(cons.mlflow_experiment_name)
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        self.assertGreater(len(runs), 0)

    def test_smoke_mlflow_run_status_finished(self):
        run = self._get_run()
        self.assertEqual(run.info.status, "FINISHED")


# ---------------------------------------------------------------------------
# Logging / params tests
# ---------------------------------------------------------------------------


class TestSmokeLogging(_SmokeBase):

    def test_smoke_mlflow_params_logged(self):
        run = self._get_run()
        expected = {"model_id", "num_epochs", "batch_size", "learning_rate",
                    "image_width", "image_height", "random_state", "device_type"}
        self.assertTrue(expected.issubset(run.data.params.keys()))

    def test_smoke_mlflow_train_loss_logged(self):
        client = mlflow.tracking.MlflowClient()
        history = client.get_metric_history(self._run_id, "train_loss")
        self.assertGreater(len(history), 0)

    def test_smoke_mlflow_train_acc_logged(self):
        client = mlflow.tracking.MlflowClient()
        history = client.get_metric_history(self._run_id, "train_acc")
        self.assertGreater(len(history), 0)

    def test_smoke_mlflow_valid_loss_logged(self):
        client = mlflow.tracking.MlflowClient()
        history = client.get_metric_history(self._run_id, "valid_loss")
        self.assertGreater(len(history), 0)

    def test_smoke_mlflow_valid_acc_logged(self):
        client = mlflow.tracking.MlflowClient()
        history = client.get_metric_history(self._run_id, "valid_acc")
        self.assertGreater(len(history), 0)


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------


class TestSmokeEvaluation(_SmokeBase):

    def test_smoke_mlflow_eval_metrics_logged(self):
        run = self._get_run()
        for key in ("test_accuracy", "test_precision", "test_recall", "test_f1"):
            self.assertIn(key, run.data.metrics, msg=f"Missing metric: {key}")

    def test_smoke_mlflow_submission_csv_artefact_present(self):
        artefacts = self._list_artefacts()
        self.assertTrue(
            any("submission.csv" in a for a in artefacts),
            msg=f"submission.csv not found in artefacts: {artefacts}",
        )


# ---------------------------------------------------------------------------
# Serialisation tests
# ---------------------------------------------------------------------------


class TestSmokeSerialization(_SmokeBase):

    def test_smoke_mlflow_pt_artefact_present(self):
        artefacts = self._list_artefacts()
        self.assertTrue(
            any(a.endswith(".pt") for a in artefacts),
            msg=f"No .pt artefact found: {artefacts}",
        )

    def test_smoke_mlflow_onnx_artefact_present(self):
        artefacts = self._list_artefacts()
        self.assertTrue(
            any(a.endswith(".onnx") for a in artefacts),
            msg=f"No .onnx artefact found: {artefacts}",
        )

    def test_smoke_onnx_artefact_is_runnable(self):
        """Download the .onnx artefact and verify it runs via onnxruntime."""
        import numpy as np
        import onnxruntime as ort

        client = mlflow.tracking.MlflowClient()
        artefacts = client.list_artifacts(self._run_id)
        onnx_artefact = next((a for a in artefacts if a.path.endswith(".onnx")), None)
        self.assertIsNotNone(onnx_artefact, "No .onnx artefact found in run")
        local_path = client.download_artifacts(self._run_id, onnx_artefact.path)
        sess = ort.InferenceSession(local_path)
        dummy = np.random.randn(2, 3, IMAGE_HEIGHT, IMAGE_WIDTH).astype(np.float32)
        out = sess.run(None, {"input": dummy})
        self.assertEqual(out[0].shape, (2, NUM_CLASSES))


if __name__ == "__main__":
    unittest.main()
