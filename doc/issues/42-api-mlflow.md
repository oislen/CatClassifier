# Issue #43 — Integrate MLflow into the Flask API

## Overview

The CatClassifier Flask API (`model/api.py`) currently classifies images with no observability — each inference request is a black box with no timing, no parameter logging, and no linkage to the model version that produced the result. This plan integrates the MLflow tracing and tagging capabilities already present in `prg_torch_model.py` into the API so that every inference request is tracked as a trace, with span-level timing, request metadata, prediction outputs, and a link back to the registered model version that served the request.

The API is stateless (one HTTP request per call) and long-running (Flask process stays alive between requests), so the integration pattern differs from the training pipeline:

- **No `mlflow.start_run`** — inference requests are not training runs and should not appear in the Experiments tab. Tracing is used instead, which operates independently of run context.
- **Per-request traces** — each POST to `/catclassifier` produces one trace entry in the Traces tab, showing the full span tree for that request.
- **Tags on spans** — request metadata (model ID, device, image filename, git commit) is attached as span inputs/attributes rather than run-level tags.

---

## Background

### Tracing without a run context

In MLflow 3.x, `mlflow.start_span` works both inside and outside an active `mlflow.start_run` context. When called outside a run, traces are recorded against the experiment set via `mlflow.set_experiment()`. This is the correct pattern for inference workloads — they produce traces but not experiment run entries.

### Flask request lifecycle

Flask processes requests sequentially (in single-threaded development mode) or concurrently (with a multi-threaded/multi-process WSGI server). Each request handler call is independent. The root span for each request must be opened and closed within the scope of the `endpoint()` function so that span context does not leak between requests.

---

## Scope

1. Configure MLflow tracking URI and experiment in `api.py` at startup
2. Instrument the `endpoint()` request handler with a root `CHAIN` span per request
3. Instrument the `classify_image()` function with child spans:
   - Model load (`TASK`)
   - Image loading and dataframe creation (`TASK`)
   - DataLoader creation (`TASK`)
   - Model prediction (`TASK`)
4. Log request metadata (model ID, device, image filename, git commit) as span inputs
5. Log prediction outputs (category, probability, response dict) as span outputs
6. Link the serving span to the registered model version via a span attribute
7. Create new unit tests for the instrumented logic
8. Update smoke tests to cover API tracing

---

## Implementation Plan

### 1. MLflow startup configuration — `model/api.py`

Add tracking configuration at module load time, before the Flask app is defined. This runs once when the Flask process starts:

```python
import mlflow
from model.utilities import get_git_commit_sha

# configure MLflow at startup — tracing only, no run context
mlflow.set_tracking_uri(cons.mlflow_tracking_uri)
mlflow.set_experiment(cons.mlflow_experiment_name)
```

`mlflow_tracking_uri` and `mlflow_experiment_name` are already defined in `cons.py` from issue #42.

---

### 2. Root request span — `endpoint()` in `model/api.py`

Wrap the body of the `endpoint()` function in a root `CHAIN` span. One trace is created per HTTP request:

```python
@app.route("/catclassifier", methods=["POST"])
def endpoint():
    logging.info("Received POST request at /catclassifier endpoint")
    if "image" not in request.files:
        return jsonify({"error": "No 'image' part in the request"}), 400

    image_file = request.files["image"]
    model_id = request.form.get("model_id", list(model_dict.keys())[1])
    device_type = request.form.get("device_type", cons.device_type)
    safe_filename = secure_filename(image_file.filename)

    with mlflow.start_span(name="api_request", span_type="CHAIN") as root_span:
        root_span.set_inputs({
            "model_id": model_id,
            "device_type": device_type,
            "image_filename": safe_filename,
            "git_commit": get_git_commit_sha(),
        })

        # ... existing validation and classification logic ...

        root_span.set_outputs({"status_code": 200})
    return jsonify(classification_result), 200
```

> `get_git_commit_sha()` is called once per request. Since git commit SHA is static for the duration of a process, consider caching it at module load time as a module-level constant to avoid repeated subprocess calls:
> ```python
> _GIT_COMMIT = get_git_commit_sha()
> ```

---

### 3. Child spans inside `classify_image()` — `model/api.py`

Instrument each stage of `classify_image()` with a named child span. These are automatically nested under the `api_request` root span because the function is called within that span's context:

```python
def classify_image(image_filepath: str, model_id: str, device_type: str) -> dict:
    device = torch.device(device_type)
    model = model_dict[model_id].to(device)

    # --- model load span ---
    with mlflow.start_span(name="model_load", span_type="TASK") as span:
        span.set_inputs({"model_id": model_id, "device_type": device_type})
        model.load(input_fpath=cons.torch_model_pt_fpath.format(model_id=model_id))
        time_logger.log_time(parent_key="ModelSerialisation", sub_key="Load")
        span.set_outputs({"loaded": True})

    # --- image loading span ---
    with mlflow.start_span(name="image_loading", span_type="TASK") as span:
        span.set_inputs({"image_filepath": image_filepath})
        torch_load_images = TorchLoadImages(torch_transforms=torch_transforms, n_workers=None)
        test_df = pd.DataFrame.from_records(
            torch_load_images.load_images(filepaths=[image_filepath])
        )
        time_logger.log_time(parent_key="DataPrep", sub_key="TrainDataLoad")
        span.set_outputs({"ndims": int(test_df["ndims"].values[0]) if len(test_df) > 0 else -1})

    # --- dataloader span ---
    with mlflow.start_span(name="dataloader_creation", span_type="TASK") as span:
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
        span.set_inputs({"model_id": model_id})
        predict = model.predict(test_loader, device)
        category = np.argmax(predict, axis=-1)
        test_df["category_id"] = category
        test_df["category_name"] = test_df["category_id"].replace(cons.category_mapper)
        test_df["probability"] = test_df["category_id"].apply(lambda x: predict[0][x])
        test_df = test_df.reset_index()
        del test_dataset
        del test_loader
        time_logger.log_time(parent_key="TestSet", sub_key="ModelPredictions")
        span.set_outputs({
            "category_id": int(category[0]),
            "category_name": str(test_df["category_name"].values[0]),
            "probability": float(test_df["probability"].values[0]),
        })

    # build response dict (existing logic unchanged)
    sub_cols = [...]
    response = test_df[sub_cols].to_dict(orient="records")[0]
    response["model_id"] = model_id
    response["datetime"] = str(datetime.datetime.now())
    return response
```

---

### 4. Registered model version linkage — `model/api.py`

Add the registered model name and version as a span attribute on the `model_load` span so that each inference trace is linked to the specific model version that served the request. The version can be fetched from the Model Registry at startup and cached:

```python
# at module load time — cache the latest registered model version per architecture
import mlflow

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
```

Then inside the `model_load` span:
```python
model_version = _get_latest_model_version(model_id)
span.set_attributes({"registered_model_name": cons.mlflow_registered_model_names.get(model_id, ""), "model_version": model_version})
```

> `mlflow_registered_model_names` is already defined in `cons.py` from issue #42's extension plan.

---

### 5. File Change Summary

| File | Change Type | Description |
|---|---|---|
| `model/api.py` | Modify | Add `import mlflow`; import `get_git_commit_sha`; set tracking URI and experiment at startup; cache `_GIT_COMMIT` and `_client`; add `_get_latest_model_version()`; wrap `endpoint()` in root `CHAIN` span; instrument `classify_image()` with `model_load`, `image_loading`, `dataloader_creation`, `prediction` child spans |

No new dependencies are required — `mlflow` is already in `pyproject.toml` from issue #42.

---

## Testing

### Manual verification

After starting the Flask API and submitting a POST request to `/catclassifier`, open the MLflow UI at `http://127.0.0.1:5000` and verify:

1. **Traces tab** — a new trace entry named `api_request` appears for each request.
2. Expanding the trace shows the span tree:
   ```
   api_request                (CHAIN)
   ├── model_load             (TASK)
   ├── image_loading          (TASK)
   ├── dataloader_creation    (TASK)
   └── prediction             (TASK)
   ```
3. The `api_request` root span inputs include `model_id`, `device_type`, `image_filename`, and `git_commit`.
4. The `prediction` span outputs include `category_id`, `category_name`, and `probability`.
5. The `model_load` span attributes include `registered_model_name` and `model_version`.
6. Each span shows its start time and duration — allowing per-stage latency analysis across requests.

### Unit tests

#### New test file: `model/unittests/test_api/test_classify_image_spans.py`

Covers the span structure produced by `classify_image()` using a mocked MLflow client and a synthetic image fixture. Tests use `unittest.mock.patch` on `mlflow.start_span` to capture span names and types without requiring a running MLflow server.

| Test name | What it verifies |
|---|---|
| `test_classify_image_creates_model_load_span` | `mlflow.start_span` is called with `name="model_load"` during `classify_image()` |
| `test_classify_image_creates_image_loading_span` | `mlflow.start_span` is called with `name="image_loading"` |
| `test_classify_image_creates_dataloader_span` | `mlflow.start_span` is called with `name="dataloader_creation"` |
| `test_classify_image_creates_prediction_span` | `mlflow.start_span` is called with `name="prediction"` |
| `test_classify_image_span_count` | Exactly 4 child spans are created per `classify_image()` call |

#### New test file: `model/unittests/test_api/test_get_latest_model_version.py`

Covers the `_get_latest_model_version()` helper using `unittest.mock.patch` on `MlflowClient`.

| Test name | What it verifies |
|---|---|
| `test_returns_string` | Return value is a `str` |
| `test_returns_version_when_model_exists` | Returns the version string when `get_registered_model` succeeds and `latest_versions` is non-empty |
| `test_returns_unknown_when_model_not_found` | Returns `"unknown"` when `get_registered_model` raises `MlflowException` |
| `test_returns_unknown_when_no_versions` | Returns `"unknown"` when `latest_versions` is an empty list |
| `test_returns_unknown_for_unrecognised_model_id` | Returns `"unknown"` when `model_id` is not in `cons.mlflow_registered_model_names` |

### Smoke tests

#### New test file: `model/smoketests/test_smoke_api_mlflow.py`

End-to-end smoke test using a file-based MLflow store and a real (or mocked) Flask test client. Uses `app.test_client()` from Flask's built-in test utilities with a tiny synthetic image.

| Test name | Scope | What it verifies |
|---|---|---|
| `test_smoke_api_trace_created` | Traces | After a POST to `/catclassifier`, at least one trace is recorded in the MLflow store |
| `test_smoke_api_root_span_name` | Traces | The root span is named `api_request` |
| `test_smoke_api_child_spans_present` | Traces | The trace contains at least 4 child spans (`model_load`, `image_loading`, `dataloader_creation`, `prediction`) |
| `test_smoke_api_prediction_span_has_outputs` | Traces | The `prediction` span has a non-empty `outputs` dict containing `category_name` |
| `test_smoke_api_root_span_has_inputs` | Traces | The `api_request` root span inputs include `model_id` and `image_filename` |

---

## Notes

- The Flask API is long-running, so `mlflow.set_tracking_uri()` and `mlflow.set_experiment()` are called once at module load time, not per request.
- `get_git_commit_sha()` spawns a subprocess. Cache it as `_GIT_COMMIT = get_git_commit_sha()` at module load time to avoid a subprocess call on every request.
- `_get_latest_model_version()` queries the Model Registry at module load time per architecture. If models are re-registered (new versions added) while the Flask process is running, the cached version will be stale. A per-request lookup is more accurate but adds latency; the cached approach is preferred unless version staleness is a concern.
- `mlflow.start_span` does not require an active `mlflow.start_run` context in MLflow 3.x — traces are logged directly against the experiment without creating a run entry. This keeps the API inference traces separate from training runs in the Experiments tab.
- The `model/unittests/test_api/` directory will need an `__init__.py` if it is to be included in the standard `unittest discover unittests` path. Follow the same pattern as `test_torch_models/` (empty `__init__.py`).
- The smoke test requires an actual `.pt` model file on disk at `cons.torch_model_pt_fpath` for the `model.load()` call to succeed. Either pre-populate this from a training run or mock `model.load()` in the smoke test setup.
- `model/api.py` imports `prg_torch_model.model_dict` and `prg_torch_model.torch_transforms` at line 10. This means importing `api.py` triggers the full `prg_torch_model.py` module-level code (model instantiation). Ensure this does not conflict with MLflow configuration set in `api.py` — both files set `mlflow.set_tracking_uri` to the same value, which is idempotent and safe.
