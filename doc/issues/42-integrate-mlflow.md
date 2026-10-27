# Issue #42 - Integrate MLflow

## Overview

Integrate [MLflow](https://mlflow.org/) into the CatClassifier model training pipeline to provide experiment tracking, model versioning, logging/tracing, ONNX model serialisation, and structured model evaluation.

MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. It provides four core components: **Tracking** (recording experiments, parameters, metrics, and artefacts), **Projects** (packaging code for reproducible runs), **Models** (a standard format for packaging and deploying models), and **Model Registry** (a centralised store for model versioning and lifecycle management). For CatClassifier, MLflow addresses a number of practical gaps in the current training pipeline. At present, each training run of `prg_torch_model.py` produces `.pt` checkpoint files and report plots on disk with no structured record of what hyperparameters were used, which model architecture was selected, or how the validation metrics evolved across epochs. Comparing the three supported architectures — AlexNet, VGG-16, and ResNet-50 — currently requires manual inspection of log output. By integrating MLflow, every run will be captured as a named experiment with full parameter provenance, per-epoch loss and accuracy curves, and linked artefacts (model weights, ONNX exports, prediction plots, and submission CSVs), making it straightforward to audit, compare, and reproduce any past training run.

**Issue:** [#42 - Integrate MlFlow](https://github.com/oislen/CatClassifier/issues/42)  
**Milestone:** Cat Classifier Version 1  
**Label:** enhancement

---

## Scope

The integration covers four areas from the ticket:

1. Model versioning and tracking
2. Logging and tracing
3. Model serialisation with ONNX
4. Model evaluation

---

## Prerequisites

### Dependency additions (`pyproject.toml`)

Add the following packages to the `dependencies` list:

```toml
"mlflow==2.x.x",
"onnx==1.x.x",
"onnxruntime==1.x.x",
```

> Pin to compatible versions with `torch==2.8.0` and `torchvision==0.23.0`. Verify with `uv lock` after updating.

### Local MLflow tracking server

Start a local MLflow server before running any training. The server must be running for `mlflow.log_params`, `mlflow.log_metrics`, and `mlflow.log_artifact` calls to succeed.

```cmd
:: from the CatClassifier root directory
uv run mlflow server --host 127.0.0.1 --port 5000
```

The UI is then accessible at `http://127.0.0.1:5000`. The server stores run data under `./mlruns/` by default; this directory should be added to `.gitignore` to avoid committing experiment artefacts.

To run the server in the background on Windows:

```cmd
start /B uv run mlflow server --host 127.0.0.1 --port 5000
```

---

## Implementation Plan

### 1. Model Versioning and Tracking

**Goal:** Wrap each training run inside an MLflow experiment and run context so that hyperparameters, architecture choices, and artefacts are recorded per run.

**Changes required:**

#### `model/cons.py`
- Add constants for the MLflow tracking URI and experiment name:
  ```python
  mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
  mlflow_experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "CatClassifier")
  ```
  The default URI points at the local MLflow server. Do **not** add `mlflow_run_id` here — it must be read at call time in `prg_torch_model.py` to avoid the value being captured at import time (see run resumption below).

#### `model/prg_torch_model.py`
- Import `mlflow` at the top of the file.
- Before training begins, set the tracking URI and experiment:
  ```python
  import mlflow
  mlflow.set_tracking_uri(cons.mlflow_tracking_uri)
  mlflow.set_experiment(cons.mlflow_experiment_name)
  ```
- Wrap **both** the `run_model_training` and `run_testset_prediction` blocks in a single `mlflow.start_run` context so that training metrics and evaluation metrics share one run. Read `MLFLOW_RUN_ID` inline at call time — not via `cons` — to ensure the environment variable is evaluated when the script runs, not at module import:
  ```python
  mlflow_run_id = os.environ.get("MLFLOW_RUN_ID", None)
  with mlflow.start_run(run_id=mlflow_run_id, run_name=model_id) as run:
      # print the active run ID so it can be captured for resumption
      logging.info(f"MLflow run_id: {run.info.run_id}")

      if input_params_dict["run_model_training"]:
          mlflow.log_params({...})
          # ... training logic ...

      if input_params_dict["run_testset_prediction"]:
          # ... prediction and evaluation logic ...
  ```
- **Run resumption pattern:** when `--run_model_training` and `--run_testset_prediction` are invoked in separate processes, the first invocation creates a new run and logs the `run_id` to stdout. The second invocation sets `MLFLOW_RUN_ID` to that value so `mlflow.start_run` reopens the same run:
  ```cmd
  :: First call — training only
  uv run python prg_torch_model.py --run_model_training --model_id alexnet_8_pretrained --device_type cpu

  :: Second call — prediction only, resuming the same run
  set MLFLOW_RUN_ID=<run_id from first call>
  uv run python prg_torch_model.py --run_testset_prediction --model_id alexnet_8_pretrained --device_type cpu
  ```
- Log hyperparameters at run start, guarded so they are only logged on a fresh run (not when resuming), since MLflow params are immutable and re-logging raises an exception:
  ```python
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
  ```

---

### 2. Logging and Tracing

**Goal:** Emit per-epoch training and validation metrics to MLflow so that training curves are visible in the MLflow UI.

**Changes required:**

#### `model/torch_models/fit.py`
- Import `mlflow` inside the function (avoids a top-level dependency for modules that do not need it).
- After each epoch's metrics are computed, add:
  ```python
  mlflow.log_metrics(
      {
          "train_loss": train_loss,
          "train_acc": train_acc,
      },
      step=epoch,
  )
  if valid_data_loader is not None:
      mlflow.log_metrics(
          {
              "valid_loss": valid_loss,
              "valid_acc": valid_acc,
          },
          step=epoch,
      )
  ```
- This fits naturally after the existing `logging.info` calls at lines 108 and 119–121.

#### `model/prg_torch_model.py`
- After the `time_logger` calls, log timing data as MLflow metrics if desired (optional enhancement). Read `time_logger.cumulative_time` directly — `TimeIt` exposes no `get_elapsed` method:
  ```python
  mlflow.log_metric("training_time_seconds", time_logger.cumulative_time)
  ```

---

### 3. Model Serialisation with ONNX

**Goal:** Export the trained PyTorch model to ONNX format as an MLflow artefact alongside the existing `.pt` checkpoint.

**Changes required:**

#### `model/cons.py`
- Add an ONNX model file path constant:
  ```python
  torch_model_onnx_fpath = os.path.join(data_models_fdir, "{model_id}.onnx")
  ```

#### `model/torch_models/export_onnx.py` (new file)
- Create a dedicated module containing the `export_onnx` function:
  ```python
  import logging

  import torch
  from beartype import beartype


  @beartype
  def export_onnx(model, output_fpath: str, image_width: int, image_height: int) -> str:
      """Exports a torch model to ONNX format.

      Parameters
      ----------
      model : CustomModelClass
          The trained torch model to export
      output_fpath : str
          The output file location to write the ONNX model to disk
      image_width : int
          The input image width expected by the model
      image_height : int
          The input image height expected by the model

      Returns
      -------
      str
          The export status message
      """
      dummy_input = torch.randn(1, 3, image_width, image_height)
      torch.onnx.export(
          model,
          dummy_input,
          output_fpath,
          export_params=True,
          opset_version=17,
          input_names=["input"],
          output_names=["output"],
          dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
      )
      msg = f"Exported ONNX model to {output_fpath}"
      logging.info(msg)
      return msg
  ```
- Export `export_onnx` from `model/torch_models/__init__.py`.

#### `model/prg_torch_model.py`
- After the existing `model.save(...)` call, add ONNX export and log both artefacts to MLflow:
  ```python
  from model.torch_models import export_onnx

  # Export to ONNX
  onnx_fpath = cons.torch_model_onnx_fpath.format(model_id=model.model_id)
  export_onnx(model, onnx_fpath, cons.IMAGE_WIDTH, cons.IMAGE_HEIGHT)

  # Log model artefacts to MLflow
  mlflow.log_artifact(cons.torch_model_pt_fpath.format(model_id=model.model_id))
  mlflow.log_artifact(onnx_fpath)
  ```

---

### 4. Model Evaluation

**Goal:** Log final test-set evaluation metrics to the MLflow run so that model quality is captured alongside training artefacts.

**Changes required:**

#### `model/utilities/` (new file `model/utilities/evaluate.py` or inline in `prg_torch_model.py`)
- Compute accuracy, precision, recall, and F1 from the test-set predictions:
  ```python
  from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

  def compute_eval_metrics(y_true, y_pred) -> dict:
      return {
          "test_accuracy": accuracy_score(y_true, y_pred),
          "test_precision": precision_score(y_true, y_pred, average="binary", pos_label="dog"),
          "test_recall": recall_score(y_true, y_pred, average="binary", pos_label="dog"),
          "test_f1": f1_score(y_true, y_pred, average="binary", pos_label="dog"),
      }
  ```

#### `model/prg_torch_model.py`
- After `test_df["category"]` is populated (line ~263), compute and log metrics:
  ```python
  # Derive ground-truth labels from filenames (cat.*.jpg -> cat, dog.*.jpg -> dog)
  test_df["true_label"] = test_df["filenames"].str.split(".").str[0]

  eval_metrics = compute_eval_metrics(test_df["true_label"], test_df["category"])
  mlflow.log_metrics(eval_metrics)
  ```
- Log the submission CSV and prediction plots as artefacts:
  ```python
  mlflow.log_artifact(cons.submission_csv_fpath)
  mlflow.log_artifact(cons.torch_pred_images_fpath)
  ```

---

## File Change Summary

| File | Change Type | Description |
|---|---|---|
| `pyproject.toml` | Modify | Add `mlflow`, `onnx`, `onnxruntime` dependencies |
| `model/cons.py` | Modify | Add MLflow URI/experiment constants, ONNX output path |
| `model/prg_torch_model.py` | Modify | Wrap both run blocks in `mlflow.start_run`, support run resumption via `run_id`, log params, artefacts, eval metrics |
| `model/torch_models/fit.py` | Modify | Log per-epoch train/validation metrics via `mlflow.log_metrics` |
| `model/torch_models/export_onnx.py` | Create | New module containing the `export_onnx` function |
| `model/torch_models/__init__.py` | Modify | Export `export_onnx` from the new module |
| `model/utilities/evaluate.py` | Create | `compute_eval_metrics` helper using `sklearn` |
| `model/utilities/__init__.py` | Modify | Export `compute_eval_metrics` from the new module |

---

## Testing

- Run `model/exeUnittests.cmd` after each change to confirm no regressions.
- Start an MLflow UI locally (`mlflow ui`) and verify:
  - Experiment and run appear after a training execution.
  - Hyperparameters are populated on the run page.
  - Per-epoch `train_loss`, `train_acc`, `valid_loss`, `valid_acc` curves render correctly.
  - `.pt` and `.onnx` artefacts are present on the run.
  - Test-set evaluation metrics (`test_accuracy`, `test_precision`, `test_recall`, `test_f1`) are logged.
- Validate the ONNX export with `onnxruntime`:
  ```python
  import onnxruntime as ort
  import numpy as np
  sess = ort.InferenceSession("path/to/model.onnx")
  dummy = np.random.randn(1, 3, 128, 128).astype(np.float32)
  out = sess.run(None, {"input": dummy})
  assert out[0].shape == (1, 2)
  ```

### Unit Tests

#### New test file: `model/unittests/test_torch_models/test_export_onnx.py`

Covers the new `export_onnx` function in `torch_models/export_onnx.py`. Uses the existing `_TinyNet` / minimal-model pattern established in `test_save_load.py` and a `tempfile` for the output path.

| Test name | What it verifies |
|---|---|
| `test_export_onnx_returns_string` | Return value of `export_onnx` is a `str` |
| `test_export_onnx_message_contains_filepath` | Returned message includes the output path |
| `test_export_onnx_creates_file` | A file is created at the specified path after the call |
| `test_export_onnx_file_is_nonempty` | The produced `.onnx` file is not empty (size > 0) |
| `test_export_onnx_file_is_valid_onnx` | File can be loaded via `onnx.load()` without raising |
| `test_export_onnx_output_shape` | Running the ONNX model via `onnxruntime` on a `(1, 3, H, W)` dummy input produces the expected `(1, num_classes)` output shape |
| `test_export_onnx_dynamic_batch_axis` | Running with batch size 4 also produces the correct output shape `(4, num_classes)`, confirming the dynamic batch axis was set |
| `test_export_onnx_overwrites_existing_file` | Calling `export_onnx` twice to the same path silently overwrites without raising |

#### New test file: `model/unittests/utilities/test_evaluate.py`

Covers the new `compute_eval_metrics` helper in `model/utilities/evaluate.py`.

| Test name | What it verifies |
|---|---|
| `test_returns_dict` | Return value is a `dict` |
| `test_dict_contains_accuracy_key` | Key `test_accuracy` is present |
| `test_dict_contains_precision_key` | Key `test_precision` is present |
| `test_dict_contains_recall_key` | Key `test_recall` is present |
| `test_dict_contains_f1_key` | Key `test_f1` is present |
| `test_key_count` | Dict contains exactly 4 keys |
| `test_perfect_predictions_accuracy` | All-correct predictions produce `test_accuracy == 1.0` |
| `test_perfect_predictions_f1` | All-correct predictions produce `test_f1 == 1.0` |
| `test_all_wrong_predictions_accuracy` | All-incorrect predictions produce `test_accuracy == 0.0` |
| `test_metric_values_are_floats` | All four metric values are `float` instances |
| `test_metric_values_in_range` | All four metric values are in `[0.0, 1.0]` |
| `test_known_mixed_accuracy` | A fixed known input/output pair produces the expected `test_accuracy` value (validates formula correctness) |
| `test_known_mixed_f1` | Same fixed pair produces the expected `test_f1` value |

#### Updates to existing test file: `model/unittests/test_torch_models/test_save_load.py`

Since `export_onnx` now lives in its own `torch_models/export_onnx.py` module and not in `save.py`, `test_save_load.py` requires no new tests. The independence of the two serialisation paths is instead covered by `test_export_onnx.py` — specifically `test_export_onnx_creates_file` confirms ONNX export only writes when explicitly called, and the `TestSave` tests already confirm `save()` writes only a `.pt` file.

#### Updates to existing test file: `model/unittests/test_torch_models/test_ModelFit.py`

`ModelFit` itself is not changed, but the per-epoch MLflow logging in `fit.py` reads directly from `ModelFit.history`. Add one test to guard against inadvertent structural changes that would break the MLflow integration:

| Test name | What it verifies |
|---|---|
| `test_history_metric_keys_are_mlflow_compatible` | All keys in `ModelFit.history` are non-empty strings containing only alphanumeric characters, underscores, hyphens, spaces, and dots — matching MLflow's metric key validation rules |

### Smoke Tests

Smoke tests exercise the full `prg_torch_model.py` pipeline end-to-end against a minimal synthetic dataset to confirm that MLflow tracking, ONNX export, and evaluation metrics are written correctly without requiring real image data or a GPU. They sit above unit tests in scope but below a full training run in cost.

#### Setup

- Set `MLFLOW_TRACKING_URI` to a `file://` URI pointing at a temporary directory (e.g. `f"file://{tempfile.mkdtemp()}"`) so runs are isolated, cleaned up after each test, and do not require the HTTP server to be running. This is intentional — smoke tests use the file-based tracking backend rather than the HTTP server, which means the HTTP transport layer is not exercised. Manual verification against the live local server (as described in the Testing section intro) covers that path.
- Set `FAST_RUN = True` and `min_epochs = 1` via monkeypatching or environment variables so the model trains for a single epoch only.
- Synthesise a small batch of random `(3, 128, 128)` image tensors with binary labels to stand in for real cat/dog images, bypassing file I/O entirely.
- Run both `--run_model_training` and `--run_testset_prediction` code paths by calling the relevant logic directly (or via `subprocess` invoking `prg_torch_model.py` with `--model_id alexnet_8_pretrained --device_type cpu`).

#### New test file: `model/unittests/test_smoke_mlflow.py`

| Test name | Scope | What it verifies |
|---|---|---|
| `test_smoke_mlflow_experiment_created` | Tracking | After a training run, an MLflow experiment with the name defined in `cons.mlflow_experiment_name` exists in the temporary tracking store |
| `test_smoke_mlflow_run_created` | Tracking | At least one run is recorded under the experiment after the pipeline executes |
| `test_smoke_mlflow_run_status_finished` | Tracking | The run's `status` is `"FINISHED"` (not `"FAILED"` or `"RUNNING"`), confirming `mlflow.start_run` exited cleanly |
| `test_smoke_mlflow_params_logged` | Logging | The run's `params` dict contains all expected keys: `model_id`, `num_epochs`, `batch_size`, `learning_rate`, `image_width`, `image_height`, `random_state`, `device_type` |
| `test_smoke_mlflow_train_loss_logged` | Logging | The run's metric history for `train_loss` is non-empty and contains at least one entry per epoch |
| `test_smoke_mlflow_train_acc_logged` | Logging | The run's metric history for `train_acc` is non-empty |
| `test_smoke_mlflow_valid_loss_logged` | Logging | The run's metric history for `valid_loss` is non-empty when a validation loader is provided |
| `test_smoke_mlflow_valid_acc_logged` | Logging | The run's metric history for `valid_acc` is non-empty when a validation loader is provided |
| `test_smoke_mlflow_eval_metrics_logged` | Evaluation | The run's metrics contain `test_accuracy`, `test_precision`, `test_recall`, and `test_f1` after the testset prediction step |
| `test_smoke_mlflow_pt_artefact_present` | Serialisation | The run's artefacts include a `.pt` file |
| `test_smoke_mlflow_onnx_artefact_present` | Serialisation | The run's artefacts include a `.onnx` file |
| `test_smoke_onnx_artefact_is_runnable` | Serialisation | The `.onnx` artefact downloaded from the run can be loaded by `onnxruntime` and produces output of the correct shape `(batch, 2)` |
| `test_smoke_mlflow_submission_csv_artefact_present` | Evaluation | The run's artefacts include `submission.csv` |

#### Running the smoke tests locally

```cmd
:: from CatClassifier/model/
uv run python -m unittest unittests.test_smoke_mlflow
```

Smoke tests are intentionally excluded from the `python-unittests` CI job in `.github/workflows/dev-pull-requests.yml` because they require more memory and wall-clock time than unit tests. They should be run manually before raising a PR for this issue, and can optionally be added to a separate `smoke-tests` CI job gated on the `main` branch workflow (`.github/workflows/deploy-main-push.yml`).

---

## Notes

- The MLflow tracking server must be running locally on `http://127.0.0.1:5000` before any training or prediction invocation. The server can be started with `uv run mlflow server --host 127.0.0.1 --port 5000` from the project root. The `mlruns/` directory it creates should be added to `.gitignore`.
- When `--run_model_training` and `--run_testset_prediction` are called in the same process invocation (as in `exeTorchModel.cmd`), both blocks share a single `mlflow.start_run` context automatically. When they are called in separate processes, pass the `run_id` logged by the first invocation via the `MLFLOW_RUN_ID` environment variable so the second call resumes the same run.
- `scikit-learn` is already a project dependency (`scikit-learn==1.8.0`), so no additional package is needed for evaluation metrics.
- The unit test file for the new `export_onnx` module (`test_export_onnx.py`) references `export_onnx` from `model/torch_models/export_onnx.py` directly, not from `save.py`. Update the existing `test_save_load.py` reference in the Unit Tests section accordingly — the `test_save_does_not_produce_onnx_file` guard test remains valid but its import path changes to `from torch_models.export_onnx import export_onnx`.
