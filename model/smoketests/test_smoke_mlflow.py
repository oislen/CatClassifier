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
import platform
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cons
import mlflow
from torch_models import EarlyStopper, ModelFit, export_onnx
from torch_models.fit import fit as fit_fn
from utilities.evaluate import compute_eval_metrics
from utilities.git_utils import get_git_commit_sha

# ---------------------------------------------------------------------------
# Minimal model shared across all smoke tests
# ---------------------------------------------------------------------------

IMAGE_WIDTH = 16
IMAGE_HEIGHT = 16
NUM_CLASSES = 2
BATCH_SIZE = 4
NUM_SAMPLES = 16
MODEL_ID = "small_net"


class _SmallNet(nn.Module):
    """Tiny conv-free net for fast CPU smoke testing."""

    def __init__(self):
        super().__init__()
        self.model_id = MODEL_ID
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
        # Use a fast sampling interval so system metrics are captured in short runs
        mlflow.enable_system_metrics_logging()
        mlflow.set_system_metrics_sampling_interval(1)
        mlflow.set_experiment(cons.mlflow_experiment_name)
        cls._run_id = cls._run_pipeline()
        # Allow system metrics background thread to flush at least one sample
        time.sleep(3)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    @classmethod
    def _run_pipeline(cls):
        """Execute a minimal training + prediction pipeline mirroring prg_torch_model.py."""
        device = torch.device("cpu")
        model = _SmallNet().to(device)
        train_loader = _make_dataloader()
        valid_loader = _make_dataloader()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        git_commit = get_git_commit_sha()

        # Create placeholder plot files so mlflow.log_artifact calls succeed
        for fname in [
            "random_image.jpg",
            "generator_plot.jpg",
            "model_accuracy.png",
            "model_loss.png",
            "pred_images.jpg",
        ]:
            fpath = os.path.join(cls._tmpdir, fname)
            with open(fpath, "wb") as f:
                f.write(b"\x00")  # minimal placeholder

        with mlflow.start_run(run_name="smoke_test") as run:
            run_id = run.info.run_id

            # (1) Tags — before root span
            mlflow.set_tags({
                "architecture": MODEL_ID,
                "device": "cpu",
                "fast_run": str(cons.FAST_RUN),
                "platform": platform.system(),
                "hostname": platform.node(),
                "git_commit": git_commit,
                "run_training": "True",
                "run_prediction": "True",
            })

            # (2) Root pipeline span
            with mlflow.start_span(name=f"pipeline:{MODEL_ID}", span_type="CHAIN") as root_span:
                root_span.set_inputs({"model_id": MODEL_ID, "device_type": "cpu"})

                mlflow.log_params({
                    "model_id": MODEL_ID,
                    "num_epochs": 1,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": 0.01,
                    "image_width": IMAGE_WIDTH,
                    "image_height": IMAGE_HEIGHT,
                    "random_state": 42,
                    "device_type": "cpu",
                })

                # --- data loading span ---
                with mlflow.start_span(name="data_loading", span_type="TASK") as span:
                    span.set_inputs({"sample_size": NUM_SAMPLES, "train_fdir": "synthetic"})
                    mlflow.log_artifact(os.path.join(cls._tmpdir, "random_image.jpg"))
                    mlflow.log_artifact(os.path.join(cls._tmpdir, "generator_plot.jpg"))
                    span.set_outputs({"df_shape": [NUM_SAMPLES, 3]})

                # --- data split span ---
                with mlflow.start_span(name="data_split", span_type="TASK") as span:
                    span.set_inputs({"total_samples": NUM_SAMPLES, "validation_frac": 0.05})
                    span.set_outputs({"train_shape": [15, 3], "validate_shape": [1, 3]})

                # --- model init span ---
                with mlflow.start_span(name="model_init", span_type="TASK") as span:
                    span.set_inputs({"model_id": MODEL_ID, "learning_rate": 0.01})
                    span.set_outputs({"device": "cpu"})

                # --- model fit span ---
                with mlflow.start_span(name="model_fit", span_type="TASK") as span:
                    span.set_inputs({"num_epochs": 1, "batch_size": BATCH_SIZE})
                    model.fit(
                        device=device,
                        criterion=criterion,
                        optimizer=optimizer,
                        train_dataloader=train_loader,
                        num_epochs=1,
                        valid_data_loader=valid_loader,
                    )
                    mlflow.log_artifact(os.path.join(cls._tmpdir, "model_accuracy.png"))
                    mlflow.log_artifact(os.path.join(cls._tmpdir, "model_loss.png"))
                    span.set_outputs({"final_train_loss": model.model_fit.history["loss"][-1]})

                # --- model serialisation span ---
                pt_path = os.path.join(cls._tmpdir, f"{MODEL_ID}.pt")
                onnx_path = os.path.join(cls._tmpdir, f"{MODEL_ID}.onnx")
                with mlflow.start_span(name="model_serialisation", span_type="TOOL") as span:
                    span.set_inputs({"pt_fpath": pt_path, "onnx_fpath": onnx_path})
                    model.save(pt_path)
                    export_onnx(model, onnx_path, IMAGE_WIDTH, IMAGE_HEIGHT)
                    mlflow.log_artifact(pt_path)
                    mlflow.log_artifact(onnx_path)
                    span.set_outputs({"pt_saved": True, "onnx_exported": True})

                # --- model registry ---
                registered_name = f"SmokeTest-{MODEL_ID}"
                artifact_path = os.path.basename(pt_path)
                artifact_store_uri = mlflow.get_artifact_uri(artifact_path)
                client = mlflow.tracking.MlflowClient()
                try:
                    client.get_registered_model(registered_name)
                except mlflow.exceptions.MlflowException:
                    client.create_registered_model(registered_name)
                model_version = client.create_model_version(
                    name=registered_name,
                    source=artifact_store_uri,
                    run_id=run_id,
                    tags={"architecture": MODEL_ID, "git_commit": git_commit, "device": "cpu"},
                )

                # --- testset prediction span ---
                test_loader = _make_dataloader()
                predict_out = model.predict(test_loader, device)
                categories = np.argmax(predict_out, axis=-1)
                pred_labels = ["dog" if c == 1 else "cat" for c in categories]
                true_labels = ["dog" if i % 2 == 1 else "cat" for i in range(len(pred_labels))]

                with mlflow.start_span(name="testset_prediction", span_type="TASK") as span:
                    span.set_inputs({"test_fdir": "synthetic"})
                    span.set_outputs({"test_df_shape": [NUM_SAMPLES, 3]})

                # --- evaluation span ---
                with mlflow.start_span(name="evaluation", span_type="TASK") as span:
                    known_classes = list(cons.category_mapper.values())
                    span.set_inputs({"known_classes": known_classes})
                    eval_metrics = compute_eval_metrics(true_labels, pred_labels)
                    mlflow.log_metrics(eval_metrics)

                    csv_path = os.path.join(cls._tmpdir, "submission.csv")
                    pd.DataFrame({"id": range(NUM_SAMPLES), "label": pred_labels}).to_csv(csv_path, index=False)
                    mlflow.log_artifact(csv_path)
                    mlflow.log_artifact(os.path.join(cls._tmpdir, "pred_images.jpg"))
                    span.set_outputs(eval_metrics)

                root_span.set_outputs({"status": "complete"})

        cls._registered_name = registered_name
        cls._model_version = model_version
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
# Tags tests
# ---------------------------------------------------------------------------


class TestSmokeTags(_SmokeBase):

    def test_smoke_mlflow_tags_logged(self):
        run = self._get_run()
        expected = {"architecture", "device", "fast_run", "platform",
                    "hostname", "git_commit", "run_training", "run_prediction"}
        self.assertTrue(
            expected.issubset(run.data.tags.keys()),
            msg=f"Missing tags: {expected - run.data.tags.keys()}",
        )

    def test_smoke_mlflow_architecture_tag_matches_model_id(self):
        run = self._get_run()
        self.assertEqual(run.data.tags.get("architecture"), run.data.params.get("model_id"))


# ---------------------------------------------------------------------------
# Traces tests
# ---------------------------------------------------------------------------


class TestSmokeTraces(_SmokeBase):

    def _get_traces(self):
        client = mlflow.tracking.MlflowClient()
        return client.search_traces(experiment_ids=[
            mlflow.get_experiment_by_name(cons.mlflow_experiment_name).experiment_id
        ])

    def test_smoke_mlflow_trace_created(self):
        traces = self._get_traces()
        self.assertGreater(len(traces), 0)

    def test_smoke_mlflow_root_span_present(self):
        traces = self._get_traces()
        root_names = [t.info.request_metadata.get("mlflow.traceInputs", "") for t in traces]
        # Check via span data — root span name starts with "pipeline:"
        found = any(
            any(s.name.startswith("pipeline:") for s in t.data.spans)
            for t in traces
        )
        self.assertTrue(found, "No root span with name starting 'pipeline:' found")

    def test_smoke_mlflow_task_spans_present(self):
        traces = self._get_traces()
        found = any(
            any(s.span_type == "TASK" for s in t.data.spans)
            for t in traces
        )
        self.assertTrue(found, "No TASK spans found in traces")

    def test_smoke_mlflow_tool_span_present(self):
        traces = self._get_traces()
        found = any(
            any(s.span_type == "TOOL" for s in t.data.spans)
            for t in traces
        )
        self.assertTrue(found, "No TOOL spans found in traces")

    def test_smoke_mlflow_span_count(self):
        traces = self._get_traces()
        # At least 8: root + data_loading, data_split, model_init, model_fit,
        # model_serialisation, testset_prediction, evaluation (+ epoch spans)
        total = sum(len(t.data.spans) for t in traces)
        self.assertGreaterEqual(total, 8, f"Expected at least 8 spans, found {total}")

    def test_smoke_mlflow_data_loading_span_has_outputs(self):
        traces = self._get_traces()
        span = next(
            (s for t in traces for s in t.data.spans if s.name == "data_loading"),
            None,
        )
        self.assertIsNotNone(span, "data_loading span not found")
        self.assertTrue(span.outputs, "data_loading span has no outputs")

    def test_smoke_mlflow_model_fit_span_has_outputs(self):
        traces = self._get_traces()
        span = next(
            (s for t in traces for s in t.data.spans if s.name == "model_fit"),
            None,
        )
        self.assertIsNotNone(span, "model_fit span not found")
        self.assertIn(
            "final_train_loss", span.outputs,
            msg=f"model_fit span outputs: {span.outputs}",
        )


# ---------------------------------------------------------------------------
# System metrics tests
# ---------------------------------------------------------------------------


class TestSmokeSystemMetrics(_SmokeBase):

    def test_smoke_system_metrics_logged(self):
        run = self._get_run()
        system_keys = [k for k in run.data.metrics if k.startswith("system/")]
        self.assertGreater(
            len(system_keys), 0,
            msg=f"No system/* metrics found. All metric keys: {list(run.data.metrics.keys())}",
        )


# ---------------------------------------------------------------------------
# Artefact tests
# ---------------------------------------------------------------------------


class TestSmokeArtefacts(_SmokeBase):

    def test_smoke_diagnostic_plots_logged(self):
        artefacts = self._list_artefacts()
        self.assertTrue(any("random_image.jpg" in a for a in artefacts),
                        msg=f"random_image.jpg not found: {artefacts}")
        self.assertTrue(any("generator_plot.jpg" in a for a in artefacts),
                        msg=f"generator_plot.jpg not found: {artefacts}")

    def test_smoke_training_curve_plots_logged(self):
        artefacts = self._list_artefacts()
        self.assertTrue(any("model_accuracy.png" in a for a in artefacts),
                        msg=f"model_accuracy.png not found: {artefacts}")
        self.assertTrue(any("model_loss.png" in a for a in artefacts),
                        msg=f"model_loss.png not found: {artefacts}")


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


# ---------------------------------------------------------------------------
# Model Registry tests
# ---------------------------------------------------------------------------


class TestSmokeModelRegistry(_SmokeBase):

    def test_smoke_model_registered(self):
        client = mlflow.tracking.MlflowClient()
        try:
            rm = client.get_registered_model(self._registered_name)
            self.assertIsNotNone(rm)
        except Exception as e:
            self.fail(f"Registered model '{self._registered_name}' not found: {e}")

    def test_smoke_model_version_linked_to_run(self):
        self.assertEqual(self._model_version.run_id, self._run_id)

    def test_smoke_model_version_has_tags(self):
        self.assertIn("architecture", self._model_version.tags)


if __name__ == "__main__":
    unittest.main()
