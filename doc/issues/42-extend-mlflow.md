# Issue #42 Extension — MLflow Traces, System Metrics, and Tags

## Overview

This plan extends the existing MLflow integration (implemented in issue #42) with four additional features visible in the MLflow 3.x UI:

1. **Tags** — attach mutable, queryable string metadata to each run so runs can be filtered, compared, and annotated in the UI without being subject to the immutability constraint of params.
2. **Traces** — instrument key pipeline functions with `@mlflow.trace` / `mlflow.start_span` so that the Traces tab shows a span tree for each training run, capturing function inputs, outputs, and timings.
3. **System metrics** — enable automatic collection of CPU, memory, GPU, disk, and network utilisation throughout the training run so that the System metrics tab is populated.
4. **Model Registry** — register trained model artefacts with the MLflow Model Registry so that the Models tab is populated, each architecture has a named registered model with version history, and model lifecycle (Staging → Production) can be managed from the UI.

None of these features require changes to the MLflow tracking server or experiment configuration established in issue #42.

---

## Background

### MLflow Tags

Tags are mutable string key-value pairs attached to a run via `mlflow.set_tags(dict)` or `mlflow.set_tag(key, value)`. They differ from params in two important ways:

- **Mutable** — tags can be updated or added on a resumed run; params cannot be re-logged once set.
- **Queryable** — the MLflow UI run list supports filtering and sorting by tag values, making it straightforward to compare runs across architectures, devices, or git commits.

Tags are set via:

```python
mlflow.set_tags({
    "key": "value",
    ...
})
```

They can be called at any point inside an active `mlflow.start_run()` context. There is no immutability restriction, so they are also safe to call on a resumed run.

MLflow reserves a set of built-in tag keys prefixed with `mlflow.*` (e.g. `mlflow.source.git.commit`, `mlflow.user`). Custom tags should use unprefixed keys or a project-specific prefix.

### MLflow Tracing

MLflow 3.x tracing is a span-based telemetry system distinct from `log_params`/`log_metrics`. A **trace** is a tree of **spans**; each span records a named operation with its start time, duration, inputs, outputs, and optional attributes. Spans are associated with the active MLflow run automatically when `mlflow.start_run()` is active.

The two instrumentation approaches are:

- **`@mlflow.trace` decorator** — wraps an entire function; captures arguments as inputs and return value as output automatically.
- **`mlflow.start_span(name, span_type, attributes)` context manager** — for code blocks that are not single functions, or when custom input/output recording is needed.

Available `SpanType` values (from `mlflow.entities.SpanType`):

`AGENT`, `CHAIN`, `CHAT_MODEL`, `EMBEDDING`, `EVALUATOR`, `GUARDRAIL`, `LLM`, `MEMORY`, `PARSER`, `RERANKER`, `RETRIEVER`, `TASK`, `TOOL`, `UNKNOWN`, `WORKFLOW`

For a machine learning training pipeline the most semantically appropriate types are:
- `CHAIN` — the overall training pipeline (root span)
- `TASK` — discrete pipeline stages (data loading, training, serialisation, evaluation)
- `TOOL` — utility operations (ONNX export, artefact logging)

The MLflow UI Traces tab shows one **trace** per invocation of a root span. Each trace expands into its child span tree. Traces are logged when the root span exits.

> **Important:** `mlflow.start_span` does **not** propagate span context across threads by default. The PyTorch `DataLoader` uses worker subprocesses/threads; span context must not be passed into worker callables.

### MLflow System Metrics

System metrics are enabled by calling `mlflow.enable_system_metrics_logging()` **before** `mlflow.start_run()`, or by passing `log_system_metrics=True` to `mlflow.start_run()`. Once enabled, a background monitor thread collects and logs the following metrics at a configurable interval (default 10 seconds):

| Collector | Metrics logged |
|---|---|
| CPU | `cpu_utilization_percentage`, `system_memory_usage_megabytes`, `system_memory_usage_percentage` |
| Disk | `disk_usage_percentage`, `disk_read_megabytes`, `disk_write_megabytes` |
| Network | `network_receive_megabytes`, `network_transmit_megabytes` |
| GPU (NVIDIA, if `pynvml` installed) | `gpu_utilization_percentage`, `gpu_memory_usage_megabytes`, `gpu_memory_usage_percentage` |
| ROCm (AMD, if available) | Similar GPU metrics |

Additional configuration functions (all optional, called before `mlflow.start_run()`):

```python
mlflow.set_system_metrics_sampling_interval(interval)   # seconds between samples, default 10
mlflow.set_system_metrics_samples_before_logging(n)     # samples to batch before writing, default 1
```

Required packages: `psutil` (CPU/disk/network/memory) — add explicitly to `pyproject.toml`. GPU metrics require `pynvml` for NVIDIA, but this is already a transitive dependency of MLflow 3.x and must **not** be pinned explicitly (see Notes).

### MLflow Model Registry

The MLflow Model Registry provides a centralised store for registering, versioning, and promoting trained models. It populates the **Models** tab in the UI. Once a model is registered:

- Each training run that produces a new `.pt` artefact can create a new **model version** under a named registered model (e.g. `CatClassifier-AlexNet8`).
- Versions can be assigned lifecycle aliases (`champion`, `challenger`) or stages for downstream consumers to reference a stable model without hard-coding a run ID.
- The full lineage from run → artefact → registered model version is preserved and queryable.

The registration API is:

```python
model_version = mlflow.register_model(
    model_uri=f"runs:/{run.info.run_id}/{artifact_path}",
    name=registered_model_name,
    tags={"architecture": model_id, "git_commit": git_commit},
)
```

- `model_uri` uses the `runs:/` scheme to reference an artefact already logged to the current run, preserving the run lineage.
- `name` is the registered model name. A new registered model is created automatically if the name does not exist.
- `tags` are attached at the model **version** level (distinct from the run-level tags set via `mlflow.set_tags`).

The three CatClassifier architectures map to three separate registered model names:

| Architecture | Registered model name |
|---|---|
| `alexnet_8_pretrained` | `CatClassifier-AlexNet8` |
| `vgg_16_pretrained` | `CatClassifier-VGG16` |
| `resnet_50_pretrained` | `CatClassifier-ResNet50` |

A `cons.py` mapping ties model IDs to their registered names, keeping `prg_torch_model.py` free of hardcoded strings.

---

## Scope

1. Add `psutil` to `pyproject.toml` (only — `pynvml` is a transitive MLflow dependency)
2. Log run tags in `prg_torch_model.py`
3. Enable system metrics logging in `prg_torch_model.py`
4. Instrument `prg_torch_model.py` with a root trace wrapping the entire run
5. Instrument key pipeline stages with child spans:
   - Data loading (`TASK`) — also log diagnostic plots as artefacts
   - Data split (`TASK`)
   - Model initialisation (`TASK`)
   - Model fit (`TASK`) — also log training curve plots as artefacts
   - Model save + ONNX export (`TOOL` — a utility operation rather than a pipeline stage)
   - Test set prediction (`TASK`)
   - Evaluation metrics (`TASK`)
6. Instrument `model/torch_models/fit.py` with a per-epoch child span
7. Register trained models with the MLflow Model Registry in `prg_torch_model.py`
8. Add registered model name mapping to `model/cons.py`
9. Update existing smoke test to assert tags, traces, system metrics, and model registration are present

---

## Implementation Plan

### 1. Dependency additions (`pyproject.toml`)

Add only `psutil` to the `dependencies` list:

```toml
"psutil==6.x.x",
```

> Pin to the latest compatible version. `psutil` provides CPU/memory/disk/network system metrics. Do **not** add `pynvml` explicitly — MLflow 3.x already depends on `nvidia-ml-py` (the canonical package, of which `pynvml` is an alias) as a transitive dependency. Adding it explicitly risks a version conflict. If no NVIDIA GPU is present, MLflow silently skips GPU metric collection regardless.

---

### 2. Run tags — `model/prg_torch_model.py`

Log tags immediately after `mlflow.start_run()` opens the run context. Tags are safe to set on both fresh and resumed runs.

```python
import subprocess  # platform is already imported in prg_torch_model.py

with mlflow.start_run(run_id=mlflow_run_id, run_name=model_id) as run:
    logging.info(f"MLflow run_id: {run.info.run_id}")

    # Derive git commit SHA (empty string if not in a git repo)
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_commit = ""

    mlflow.set_tags({
        "architecture": model_id,
        "device": device_type,
        "fast_run": str(cons.FAST_RUN),
        "platform": platform.system(),
        "hostname": platform.node(),
        "git_commit": git_commit,
        "run_training": str(input_params_dict["run_model_training"]),
        "run_prediction": str(input_params_dict["run_testset_prediction"]),
    })
```

**Tag reference:**

| Tag key | Example value | Description |
|---|---|---|
| `architecture` | `alexnet_8_pretrained` | Model architecture used in the run |
| `device` | `cuda` | Device the model was trained on |
| `fast_run` | `False` | Whether `FAST_RUN` mode was active (debug/CI flag) |
| `platform` | `Windows` | Operating system |
| `hostname` | `my-machine` | Machine hostname (useful for multi-machine comparisons) |
| `git_commit` | `a3f9c12` | Short git commit SHA for reproducibility |
| `run_training` | `True` | Whether the training block was executed in this invocation |
| `run_prediction` | `True` | Whether the prediction block was executed in this invocation |

> `platform` and `subprocess` are already in the Python standard library — no new dependencies required.

---

### 3. System metrics — `model/prg_torch_model.py`

Enable system metrics **before** `mlflow.start_run()` so the background monitor starts with the run:

```python
# Enable system metrics collection (CPU, memory, disk, network, GPU if pynvml present)
mlflow.enable_system_metrics_logging()
mlflow.set_system_metrics_sampling_interval(10)   # sample every 10 seconds (default)

mlflow_run_id = os.environ.get("MLFLOW_RUN_ID", None)
with mlflow.start_run(run_id=mlflow_run_id, run_name=model_id) as run:
    ...
```

The `log_system_metrics=True` parameter on `mlflow.start_run()` is an alternative to the global call, but the global call is preferred here because it also covers the case where `mlflow_run_id` resumes an existing run — the global flag ensures the monitor always starts.

---

### 4. Root trace — `model/prg_torch_model.py`

Wrap the entire `with mlflow.start_run(...) as run:` block in a root span named after the model so the Traces tab shows one trace per training invocation:

```python
with mlflow.start_run(run_id=mlflow_run_id, run_name=model_id) as run:
    logging.info(f"MLflow run_id: {run.info.run_id}")

    with mlflow.start_span(name=f"pipeline:{model_id}", span_type="CHAIN") as root_span:
        root_span.set_inputs({"model_id": model_id, "device_type": device_type})

        if input_params_dict["run_model_training"]:
            # ... training spans (see below) ...

        if input_params_dict["run_testset_prediction"]:
            # ... prediction spans (see below) ...

        root_span.set_outputs({"status": "complete"})
```

---

### 5. Pipeline stage spans — `model/prg_torch_model.py`

Wrap each major stage in a named child span using `mlflow.start_span` as a context manager. All child spans created inside the root span context are automatically nested under it.

#### Training stage spans (inside `run_model_training` block)

```python
with mlflow.start_span(name="data_loading", span_type="TASK") as span:
    span.set_inputs({"sample_size": sample_size, "train_fdir": cons.train_fdir})
    # ... existing image loading, dataframe creation, and diagnostic plot calls ...
    # Log the diagnostic plots produced during this stage so they appear in the
    # run's Artefacts tab alongside the model artefacts
    mlflow.log_artifact(cons.torch_random_image_fpath)
    mlflow.log_artifact(cons.torch_generator_plot_fpath)
    span.set_outputs({"df_shape": list(df.shape)})

with mlflow.start_span(name="data_split", span_type="TASK") as span:
    span.set_inputs({"total_samples": df.shape[0], "validation_frac": 0.05})
    # ... existing train/validation split ...
    # set_outputs must be called here, before the del train_df / del validate_df
    # statements that follow in the data loaders section
    span.set_outputs({"train_shape": list(train_df.shape), "validate_shape": list(validate_df.shape)})

with mlflow.start_span(name="model_init", span_type="TASK") as span:
    span.set_inputs({"model_id": model_id, "learning_rate": cons.learning_rate})
    # ... existing criterion/optimizer/scheduler/early_stopper setup ...
    span.set_outputs({"device": str(device)})

with mlflow.start_span(name="model_fit", span_type="TASK") as span:
    span.set_inputs({"num_epochs": num_epochs, "batch_size": cons.batch_size})
    # ... existing model.fit() call and plot_model_fit() call ...
    # Log the training curve plots produced by plot_model_fit() so they appear
    # in the Artefacts tab; filenames confirmed from model/utilities/plot_model.py
    mlflow.log_artifact(os.path.join(cons.torch_report_fdir, "model_accuracy.png"))
    mlflow.log_artifact(os.path.join(cons.torch_report_fdir, "model_loss.png"))
    span.set_outputs({"final_train_loss": model.model_fit.history["loss"][-1]})

with mlflow.start_span(name="model_serialisation", span_type="TOOL") as span:
    span.set_inputs({"pt_fpath": pt_fpath, "onnx_fpath": onnx_fpath})
    # ... existing model.save() and export_onnx() calls ...
    span.set_outputs({"pt_saved": True, "onnx_exported": True})
```

#### Prediction stage spans (inside `run_testset_prediction` block)

```python
with mlflow.start_span(name="testset_prediction", span_type="TASK") as span:
    span.set_inputs({"test_fdir": cons.test_fdir})
    # ... existing load/predict/replace logic ...
    span.set_outputs({"test_df_shape": list(test_df.shape)})

with mlflow.start_span(name="evaluation", span_type="TASK") as span:
    span.set_inputs({"known_classes": list(known_classes)})
    # ... existing compute_eval_metrics and mlflow.log_metrics calls ...
    # metrics_available is a boolean set just before this span based on
    # whether test filenames contain class labels (the issubset guard):
    #   metrics_available = set(test_df["true_label"].unique()).issubset(known_classes)
    span.set_outputs(eval_metrics if metrics_available else {"skipped": True})
```

---

### 6. Per-epoch span — `model/torch_models/fit.py`

Wrap each training epoch in a child span so the Traces tab shows per-epoch timing. This span sits inside the `model_fit` parent span established in `prg_torch_model.py`:

```python
for epoch in range(start_epoch, end_epoch):
    with mlflow.start_span(name=f"epoch_{epoch}", span_type="TASK") as epoch_span:
        epoch_span.set_inputs({"epoch": epoch, "num_epochs": num_epochs})
        # ... existing forward pass, backward pass, metric calculation ...
        epoch_span.set_outputs({
            "train_loss": train_loss,
            "train_acc": train_acc,
        })
```

> **Note:** Do not pass span context into `DataLoader` worker threads. The per-epoch span must only wrap code that runs in the main process thread (the epoch loop body after the dataloader iteration is complete).

---

### 7. Model Registry — `model/cons.py` and `model/prg_torch_model.py`

#### `model/cons.py`

Add a mapping from model ID to registered model name so the string is defined in one place:

```python
# MLflow Model Registry — registered model names per architecture
mlflow_registered_model_names = {
    "alexnet_8_pretrained": "CatClassifier-AlexNet8",
    "vgg_16_pretrained": "CatClassifier-VGG16",
    "resnet_50_pretrained": "CatClassifier-ResNet50",
}
```

#### `model/prg_torch_model.py`

After `model.save()` and `export_onnx()` have completed and both artefacts have been logged via `mlflow.log_artifact()`, register the `.pt` model with the Model Registry. Registration runs whenever the training block executes — on both fresh and resumed runs — since a resumed training run represents a genuine new version of the model weights. The only guard needed is that the training block ran (`input_params_dict["run_model_training"]` is `True`), which is already the case because this code sits inside that block. Note that `git_commit` is available here because it is assigned at the start of the `mlflow.start_run` context in Section 2, before this code executes:

```python
logging.info("Registering model with MLflow Model Registry...")
registered_name = cons.mlflow_registered_model_names[model_id]  # use model_id, not model.model_id
# artifact_path must match the path used in mlflow.log_artifact
artifact_path = os.path.basename(pt_fpath)
model_version = mlflow.register_model(
    model_uri=f"runs:/{run.info.run_id}/{artifact_path}",
    name=registered_name,
    tags={
        "architecture": model_id,
        "git_commit": git_commit,
        "device": device_type,
    },
)
logging.info(
    f"Registered model '{registered_name}' version {model_version.version}"
)
```

> **Note on `artifact_path`:** `mlflow.log_artifact(pt_fpath)` logs the file at the root of the run's artefact store, so the `artifact_path` in the `runs:/` URI is just the filename (e.g. `alexnet_8_pretrained.pt`). If artefacts are logged into a subdirectory via `mlflow.log_artifact(pt_fpath, artifact_path="models")`, the URI must be updated to `runs:/<run_id>/models/alexnet_8_pretrained.pt`.

> **Note on `await_registration_for`:** The default is 300 seconds. For a local MLflow server this resolves almost immediately; leave the default.

#### Model lifecycle (post-registration, via UI or CLI)

After registration, model versions can be promoted using the MLflow UI's Model Registry tab or via the client API. This is a manual step performed after evaluating run results — it is not automated in `prg_torch_model.py`:

```python
# Promote a version to champion alias (example, not added to prg_torch_model.py)
client = mlflow.tracking.MlflowClient()
client.set_registered_model_alias(
    name="CatClassifier-AlexNet8",
    alias="champion",
    version=model_version.version,
)
```

---

### 8. File Change Summary

| File | Change Type | Description |
|---|---|---|
| `pyproject.toml` | Modify | Add `psutil` dependency only (`pynvml` is a transitive dependency of MLflow) |
| `model/cons.py` | Modify | Add `mlflow_registered_model_names` mapping |
| `model/prg_torch_model.py` | Modify | Add `import subprocess`; log run tags; enable system metrics; add root `CHAIN` span and per-stage `TASK`/`TOOL` child spans; log diagnostic and training curve plot artefacts; register model with Model Registry |
| `model/torch_models/fit.py` | Modify | Add per-epoch `TASK` child span |
| `model/utilities/git_utils.py` | Create *(recommended)* | Extract `get_git_commit_sha()` helper from `prg_torch_model.py` to enable unit testing |
| `model/utilities/__init__.py` | Modify *(if above)* | Export `get_git_commit_sha` |
| `model/unittests/utilities/test_cons_mlflow.py` | Create | Unit tests for `cons.mlflow_registered_model_names` |
| `model/unittests/utilities/test_git_commit.py` | Create *(if extracted)* | Unit tests for `get_git_commit_sha()` using `unittest.mock.patch` |
| `model/unittests/utilities/test_plot_model.py` | Create | Unit tests verifying `plot_model_fit` writes `model_accuracy.png` and `model_loss.png` |
| `model/smoketests/test_smoke_mlflow.py` | Modify | Add 7 new smoke tests for tags, traces, system metrics, artefacts, and Model Registry |

---

## Testing

### Manual verification

After a training run, open the MLflow UI at `http://127.0.0.1:5000` and verify:

1. **Overview tab** — run tags (`architecture`, `device`, `git_commit`, etc.) appear in the Tags section.
2. **System metrics tab** is populated with CPU, memory, disk, and network charts. If running on CUDA with `pynvml` installed, GPU utilisation and GPU memory charts should also appear.
3. **Traces tab** contains one trace entry named `pipeline:<model_id>`.
4. Expanding the trace shows a span tree with the following structure:
   ```
   pipeline:alexnet_8_pretrained  (CHAIN)
   ├── data_loading               (TASK)
   ├── data_split                 (TASK)
   ├── model_init                 (TASK)
   ├── model_fit                  (TASK)
   │   ├── epoch_1                (TASK)
   │   ├── epoch_2                (TASK)
   │   └── ...
   ├── model_serialisation        (TOOL)
   ├── testset_prediction         (TASK)
   └── evaluation                 (TASK)
   ```
5. Each span shows its inputs, outputs, start time, and duration.
6. Spans with errors (e.g. if ONNX export fails) should show `ERROR` status with the exception message.
7. **Artifacts tab** — the following files should be present on the run:
   - `random_image.jpg` — sample training image (logged during `data_loading`)
   - `generator_plot.jpg` — data loader sample grid (logged during `data_loading`)
   - `model_accuracy.png` — training/validation accuracy curve (logged during `model_fit`)
   - `model_loss.png` — training/validation loss curve (logged during `model_fit`)
   - `<model_id>.pt` — serialised PyTorch weights
   - `<model_id>.onnx` — ONNX export
   - `submission.csv` — Kaggle submission file
   - `pred_images.jpg` — test set prediction sample
8. **Models tab** — a registered model named `CatClassifier-AlexNet8` (or the relevant architecture name) appears with at least version 1. Clicking through to the version shows the source run linkage and version-level tags.

### Unit tests

Most of the new logic in this extension lives inside MLflow API calls (`mlflow.set_tags`, `mlflow.start_span`, `mlflow.register_model`) which are integration-level and best covered by the smoke test. However, three areas have isolatable logic that can and should be unit-tested independently.

#### New test file: `model/unittests/utilities/test_cons_mlflow.py`

Covers the new MLflow constants added to `model/cons.py` — specifically `mlflow_registered_model_names`. These are pure data and can be tested without MLflow running.

| Test name | What it verifies |
|---|---|
| `test_registered_model_names_is_dict` | `cons.mlflow_registered_model_names` is a `dict` |
| `test_registered_model_names_contains_all_architectures` | All three model IDs (`alexnet_8_pretrained`, `vgg_16_pretrained`, `resnet_50_pretrained`) are present as keys |
| `test_registered_model_names_key_count` | The dict contains exactly 3 entries — no missing or extra architectures |
| `test_registered_model_names_values_are_strings` | All values are non-empty strings |
| `test_registered_model_names_follow_naming_convention` | All values match the pattern `CatClassifier-<Name>` (start with `"CatClassifier-"`) |
| `test_registered_model_names_no_spaces` | No registered name contains whitespace (MLflow rejects names with leading/trailing spaces) |
| `test_alexnet_registered_name` | `alexnet_8_pretrained` maps to `"CatClassifier-AlexNet8"` |
| `test_vgg16_registered_name` | `vgg_16_pretrained` maps to `"CatClassifier-VGG16"` |
| `test_resnet50_registered_name` | `resnet_50_pretrained` maps to `"CatClassifier-ResNet50"` |

#### New test file: `model/unittests/utilities/test_git_commit.py`

The `subprocess` git commit derivation is new logic added to `prg_torch_model.py`. It is worth extracting into a helper function in `model/utilities/` (e.g. `get_git_commit_sha()`) so it can be unit-tested with `unittest.mock.patch`. If kept inline in `prg_torch_model.py`, these tests are not feasible — extraction is recommended.

| Test name | What it verifies |
|---|---|
| `test_returns_string` | Return value is a `str` |
| `test_returns_short_sha_in_git_repo` | When `subprocess.check_output` returns a valid SHA, the function returns a non-empty stripped string |
| `test_returns_empty_string_when_not_git_repo` | When `subprocess.check_output` raises `subprocess.CalledProcessError`, the function returns `""` |
| `test_returns_empty_string_on_file_not_found` | When `git` is not installed (`FileNotFoundError`), the function returns `""` |
| `test_sha_has_no_leading_trailing_whitespace` | The returned string has no leading or trailing whitespace |

> If `get_git_commit_sha()` is extracted to `model/utilities/git_utils.py`, add it to `model/utilities/__init__.py` and list `model/utilities/git_utils.py` as a new `Create` entry in the File Change Summary.

#### Updates to existing test file: `model/unittests/utilities/test_plot_model.py` (new file)

`plot_model_fit` in `model/utilities/plot_model.py` writes `model_accuracy.png` and `model_loss.png` to the specified output directory. The plan's `mlflow.log_artifact` calls depend on these exact filenames. A regression test guards against a rename breaking the artefact logging silently.

| Test name | What it verifies |
|---|---|
| `test_plot_model_fit_creates_accuracy_file` | `plot_model_fit(model_fit, output_fdir=tmp_dir)` creates `model_accuracy.png` in the output directory |
| `test_plot_model_fit_creates_loss_file` | `plot_model_fit(model_fit, output_fdir=tmp_dir)` creates `model_loss.png` in the output directory |
| `test_plot_model_fit_creates_exactly_two_files` | Exactly 2 `.png` files are created — no unexpected extras (requires `tearDown` to use `shutil.rmtree` on the temp directory to ensure a clean state between tests) |
| `test_plot_model_fit_no_output_when_fdir_is_none` | When `output_fdir=None`, the function does not raise and writes no files — verified by checking neither `model_accuracy.png` nor `model_loss.png` exists at any expected path |

> Use a `tempfile.mkdtemp()` as `output_fdir` and clean up in `tearDown`. Pass a synthetic `ModelFit` fixture (identical to those in `test_ModelFit.py`) as the `model_fit` argument.

### Smoke tests

#### Updates to `model/smoketests/test_smoke_mlflow.py`

| Test name | Scope | What it verifies |
|---|---|---|
| `test_smoke_mlflow_tags_logged` | Tags | The run's tags dict contains all expected keys: `architecture`, `device`, `fast_run`, `platform`, `hostname`, `git_commit`, `run_training`, `run_prediction` |
| `test_smoke_mlflow_architecture_tag_matches_model_id` | Tags | The `architecture` tag value matches the `model_id` param that was logged |
| `test_smoke_mlflow_trace_created` | Traces | At least one trace is recorded for the run in the MLflow tracking store |
| `test_smoke_mlflow_root_span_present` | Traces | The trace contains a root span whose name starts with `"pipeline:"` |
| `test_smoke_mlflow_task_spans_present` | Traces | The trace contains at least one child span of type `"TASK"` |
| `test_smoke_mlflow_tool_span_present` | Traces | The trace contains at least one child span of type `"TOOL"` (covers `model_serialisation`) |
| `test_smoke_mlflow_span_count` | Traces | The trace contains at least 8 spans (1 root + 7 stage spans: `data_loading`, `data_split`, `model_init`, `model_fit`, `model_serialisation`, `testset_prediction`, `evaluation`) excluding per-epoch children |
| `test_smoke_mlflow_data_loading_span_has_outputs` | Traces | The `data_loading` span has a non-empty `outputs` dict |
| `test_smoke_mlflow_model_fit_span_has_outputs` | Traces | The `model_fit` span has a non-empty `outputs` dict containing `final_train_loss` |
| `test_smoke_system_metrics_logged` | System metrics | The run's metric keys include at least one system metric (e.g. `system/cpu_utilization_percentage`) |
| `test_smoke_diagnostic_plots_logged` | Artefacts | The run's artefacts include `random_image.jpg` and `generator_plot.jpg`. **Note:** the smoke test's synthetic pipeline does not invoke `plot_image` or `plot_generator`, so these files will not be created on disk. The smoke test must create synthetic placeholder image files at `cons.torch_random_image_fpath` and `cons.torch_generator_plot_fpath` in `setUpClass` before logging, or assert that `mlflow.log_artifact` was called with those paths using `unittest.mock.patch`. |
| `test_smoke_training_curve_plots_logged` | Artefacts | The run's artefacts include `model_accuracy.png` and `model_loss.png` |
| `test_smoke_model_registered` | Model Registry | A registered model exists in the tracking store whose name matches `cons.mlflow_registered_model_names[model_id]` |
| `test_smoke_model_version_linked_to_run` | Model Registry | The registered model version's `run_id` matches the run ID of the pipeline execution |
| `test_smoke_model_version_has_tags` | Model Registry | The registered model version has at least one tag (`architecture`) set |

---

## Notes

- Tags use only standard library modules (`platform`, `subprocess`) — no additional dependencies are needed for the tagging feature.
- Tags are mutable and can be added or updated on a resumed run without restriction, unlike params. The `run_training` and `run_prediction` tags accurately reflect which blocks ran in each invocation, which is particularly useful when training and prediction are split across two separate process calls.
- If the working directory is not a git repository (e.g. in CI or Docker), `git_commit` will be an empty string rather than raising an error.
- Model registration (`mlflow.register_model`) runs whenever the training block executes, whether the run is fresh or resumed. A resumed training run produces genuinely new model weights and warrants a new version. The only guard is that the training block ran — a prediction-only resumed run (where `run_model_training` is `False`) will not trigger registration because the registration call sits inside the `run_model_training` block.
- The MLflow Model Registry is fully supported with a file-based tracking store (the `file://` URI used by the smoke tests) since MLflow 2.x — a database-backed store is not required. The smoke tests can therefore exercise Model Registry assertions without any additional server setup.
- The `artifact_path` in the `runs:/` URI must exactly match the path used when the artefact was logged. If `mlflow.log_artifact(pt_fpath)` is used without a subdirectory argument, the path is just the filename. Verify this is consistent before implementing.
- Model lifecycle management (assigning `champion`/`challenger` aliases, transitioning to Production) is a manual step performed after reviewing run results — it is not automated in `prg_torch_model.py` and is not part of this ticket's scope.
- System metrics are collected on a background thread and written to the tracking server at each sampling interval. For very short runs (e.g. unit tests or smoke tests with a single epoch on a tiny dataset), the run may finish before the first sample is written. The smoke test for system metrics should therefore add a short `time.sleep(12)` after the run completes before querying, or increase the sample rate to 1 second with `mlflow.set_system_metrics_sampling_interval(1)` in the test setup.
- The `mlflow.start_span` context manager does not propagate span context across threads. The `DataLoader` worker processes are unaffected — spans should only wrap the main-thread logic, not the worker callables.
- Resuming a run (via `MLFLOW_RUN_ID`) will append new traces to the existing run. Each new invocation creates a new root span, so training-only and prediction-only runs each produce their own trace entry under the same run.
- `pynvml` (`nvidia-ml-py`) is a transitive dependency of MLflow 3.x and must **not** be pinned explicitly in `pyproject.toml` to avoid version conflicts. If no NVIDIA GPU is present, MLflow silently skips GPU metric collection. Only `psutil` needs to be added explicitly as it is not pulled in transitively.
