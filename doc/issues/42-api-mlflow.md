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
2. Cache `_GIT_COMMIT`, `_client`, and `_MODEL_VERSIONS` at module load time
3. Instrument the `endpoint()` request handler with a root `CHAIN` span per request, including `request_id` from headers
4. Add structured error handling inside `classify_image()` so exceptions are surfaced as span errors and return a JSON 500 response
5. Instrument the `classify_image()` function with child spans:
   - Model load (`TASK`) — with registered model version attributes
   - Image loading and dataframe creation (`TASK`) — with original filename
   - DataLoader creation (`TASK`)
   - Model prediction (`TASK`)
6. Log the full classification result dict as root span output
7. Link the serving span to the registered model version via span attributes
8. Create new unit tests for the instrumented logic
9. Update smoke tests to cover API tracing, including error-path traces

---

## Implementation Plan

### 1. MLflow startup configuration — `model/api.py`

Add tracking configuration, module-level caches, and helper functions at module load time, before the Flask app is defined. This runs once when the Flask process starts:

```python
import mlflow
from model.utilities import get_git_commit_sha

# configure MLflow at startup — tracing only, no run context
mlflow.set_tracking_uri(cons.mlflow_tracking_uri)
mlflow.set_experiment(cons.mlflow_experiment_name)

# cache git SHA — avoids a subprocess call on every request
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
    mid: _get_latest_model_version(mid)
    for mid in cons.mlflow_registered_model_names
}
```

`mlflow_tracking_uri`, `mlflow_experiment_name`, and `mlflow_registered_model_names` are all defined in `cons.py` from the prerequisite plan (42-extend-mlflow.md).

---

### 2. Root request span — `endpoint()` in `model/api.py`

Wrap the body of the `endpoint()` function in a root `CHAIN` span. The span must open **after** all pre-flight validation guards that return early with no image or an empty filename (before any meaningful work has started), but **before** the `model_id not in model_dict` check and file save — those should run inside the span so that invalid model ID errors and file-save failures appear as `ERROR` spans in the Traces tab (providing full observability for bad requests):

```python
@app.route("/catclassifier", methods=["POST"])
def endpoint():
    logging.info("Received POST request at /catclassifier endpoint")
    # early exits before the span — no image data yet
    if "image" not in request.files:
        return jsonify({"error": "No 'image' part in the request"}), 400

    image_file = request.files["image"]
    model_id = request.form.get("model_id", list(model_dict.keys())[1])
    device_type = request.form.get("device_type", cons.device_type)
    safe_filename = secure_filename(image_file.filename)

    if safe_filename == "":
        return jsonify({"error": "No file selected for uploading"}), 400

    # open root span — all remaining logic (including model_id validation) runs inside
    with mlflow.start_span(name="api_request", span_type="CHAIN") as root_span:
        root_span.set_inputs({
            "model_id": model_id,
            "device_type": device_type,
            "image_filename": safe_filename,
            "git_commit": _GIT_COMMIT,
            "request_id": request.headers.get("X-Request-ID", "none"),
        })

        if model_id not in model_dict:
            root_span.set_outputs({"status_code": 400, "error": "invalid model_id"})
            return (
                jsonify({"error": f"Invalid model_id. Available model_ids: {list(model_dict.keys())}"}),
                400,
            )

        logging.info("Processing the uploaded image")
        api_filepath = os.path.join(cons.api_fdir, safe_filename)
        if not os.path.exists(cons.api_fdir):
            os.makedirs(os.path.dirname(api_filepath))
        image_file.save(api_filepath)

        classification_result = classify_image(
            image_filepath=api_filepath,
            model_id=model_id,
            device_type=device_type,
            original_filename=safe_filename,
        )
        # include the full result dict alongside status code for rich trace output
        root_span.set_outputs({"status_code": 200, **classification_result})

    return jsonify(classification_result), 200
```

> **Span status on error-path returns:** a `return` statement inside a `with mlflow.start_span(...)` block exits the context manager cleanly — the span status is recorded as `OK`, not `ERROR`. Only an unhandled exception propagating out of the span context causes `ERROR` status. This means the invalid `model_id` path (which uses an early `return`) produces a span with `OK` status but with outputs `{"status_code": 400, "error": "invalid model_id"}` — visible in the Traces tab. The `ERROR` status only applies to the 500 path where `classify_image()` raises an exception.

---

### 3. Child spans and error handling inside `classify_image()` — `model/api.py`

Instrument each stage of `classify_image()` with a named child span. These are automatically nested under the `api_request` root span because the function is called within that span's context. Wrap the entire function body in a `try/except` so that unexpected errors (e.g. corrupt image, missing weights) are logged to the span and returned as a structured JSON 500 response rather than causing an unhandled Flask exception:

```python
def classify_image(image_filepath: str, model_id: str, device_type: str, original_filename: str = "") -> dict:
    device = torch.device(device_type)
    model = model_dict[model_id].to(device)

    try:
        # --- model load span ---
        with mlflow.start_span(name="model_load", span_type="TASK") as span:
            span.set_inputs({"model_id": model_id, "device_type": device_type})
            span.set_attributes({
                "registered_model_name": cons.mlflow_registered_model_names.get(model_id, ""),
                "model_version": _MODEL_VERSIONS.get(model_id, "unknown"),
            })
            model.load(input_fpath=cons.torch_model_pt_fpath.format(model_id=model_id))
            time_logger.log_time(parent_key="ModelSerialisation", sub_key="Load")
            span.set_outputs({"loaded": True})

        # --- image loading span ---
        with mlflow.start_span(name="image_loading", span_type="TASK") as span:
            span.set_inputs({
                "image_filepath": image_filepath,
                "original_filename": original_filename,  # uploaded filename for debugging
            })
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

    except Exception as exc:
        logging.exception(f"classify_image failed: {exc}")
        raise  # re-raise so the root span in endpoint() marks itself ERROR and returns 500

    # build response dict (existing logic unchanged)
    sub_cols = [...]
    response = test_df[sub_cols].to_dict(orient="records")[0]
    response["model_id"] = model_id
    response["datetime"] = str(datetime.datetime.now())
    return response
```

The `except` block re-raises the exception. The caller (`endpoint()`) must catch it and return a JSON 500:

```python
        try:
            classification_result = classify_image(
                image_filepath=api_filepath,
                model_id=model_id,
                device_type=device_type,
                original_filename=safe_filename,
            )
        except Exception as exc:
            root_span.set_outputs({"status_code": 500, "error": str(exc)})
            return jsonify({"error": "Internal classification error", "detail": str(exc)}), 500

        root_span.set_outputs({"status_code": 200, **classification_result})
```

> The `classify_image` signature gains an `original_filename` parameter. Pass `safe_filename` from `endpoint()` as shown above.

> **Implementation note:** the `try/except` block shown in this section replaces the plain `classify_image(...)` call at the end of the Section 2 `endpoint()` snippet. The full `endpoint()` function combines Section 2 (span open, inputs, model ID guard, file save) with Section 3 (try/except wrapping `classify_image`, error output, success output).

---

### 4. File Change Summary

| File | Change Type | Description |
|---|---|---|
| `model/api.py` | Modify | Add `import mlflow`; import `get_git_commit_sha`; set tracking URI and experiment at startup; cache `_GIT_COMMIT`, `_client`, `_MODEL_VERSIONS`; add `_get_latest_model_version()`; wrap `endpoint()` in root `CHAIN` span with `request_id`, full result output, and error handling; add `original_filename` parameter to `classify_image()`; instrument all four stages with child spans; surface exceptions as JSON 500 with span `ERROR` status |

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
3. The `api_request` root span inputs include `model_id`, `device_type`, `image_filename`, `git_commit`, and `request_id`.
4. The `api_request` root span outputs include `status_code`, `category_name`, `probability`, and `model_id` from the classification result.
5. The `image_loading` span inputs include both `image_filepath` (disk path) and `original_filename` (uploaded name).
6. The `prediction` span outputs include `category_id`, `category_name`, and `probability`.
7. The `model_load` span attributes include `registered_model_name` and `model_version`.
8. Each span shows its start time and duration — allowing per-stage latency analysis across requests.
9. Submit a POST with an invalid `model_id` — a trace with **`OK` status** should appear (early `return` exits the span cleanly) with outputs `{"status_code": 400, "error": "invalid model_id"}` and the HTTP response should be a 400 JSON error.
10. Simulate a classification failure (e.g. remove the `.pt` file) — a trace with **`ERROR` status** should appear (exception propagates through the span context) and the HTTP response should be a 500 JSON error.

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
| `test_classify_image_loading_span_includes_original_filename` | The `image_loading` span inputs contain the `original_filename` key |

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

End-to-end smoke test using a file-based MLflow store and Flask's built-in test client (`app.test_client()`). Mocks `model.load()` so no `.pt` file is needed. Uses a tiny synthetic JPEG-formatted bytes object as the uploaded image.

| Test name | Scope | What it verifies |
|---|---|---|
| `test_smoke_api_trace_created` | Traces | After a POST to `/catclassifier`, at least one trace is recorded in the MLflow store |
| `test_smoke_api_root_span_name` | Traces | The root span is named `api_request` |
| `test_smoke_api_child_spans_present` | Traces | The trace contains at least 4 child spans (`model_load`, `image_loading`, `dataloader_creation`, `prediction`) |
| `test_smoke_api_prediction_span_has_outputs` | Traces | The `prediction` span has a non-empty `outputs` dict containing `category_name` |
| `test_smoke_api_root_span_has_inputs` | Traces | The `api_request` root span inputs include `model_id`, `image_filename`, and `request_id` |
| `test_smoke_api_root_span_has_full_result_outputs` | Traces | The `api_request` root span outputs include `status_code`, `category_name`, and `probability` |
| `test_smoke_api_error_request_traced` | Traces | A POST with an invalid `model_id` produces a trace with `OK` status (early return, not an exception) and outputs containing `{"status_code": 400}`, and returns HTTP 400 |

---

## Notes

- **Prerequisite:** this plan requires **42-extend-mlflow.md** to have been implemented first. It depends on `cons.mlflow_registered_model_names`, `cons.mlflow_tracking_uri`, `cons.mlflow_experiment_name`, and `model.utilities.get_git_commit_sha` — all introduced in that plan.
- The Flask API is long-running, so `mlflow.set_tracking_uri()` and `mlflow.set_experiment()` are called once at module load time, not per request.
- `get_git_commit_sha()` spawns a subprocess. It is cached as `_GIT_COMMIT` at module load time to avoid a subprocess call on every request.
- `_MODEL_VERSIONS` is built at module load time by calling `_get_latest_model_version()` for each architecture. If models are re-registered while the Flask process is running, the cached versions will be stale until the process restarts. This is the agreed approach — a per-request lookup is more accurate but adds Registry query latency to every inference call.
- `classify_image()` gains an `original_filename: str = ""` parameter. The default empty string ensures backward compatibility with any existing callers that do not pass the argument.
- `request_id` is read from the `X-Request-ID` header, which is a de facto standard used by API gateways (AWS ALB, Nginx, etc.). If no header is present, `"none"` is stored. This costs nothing and enables cross-referencing MLflow traces with upstream access logs when a gateway injects the header.
- `test_loader` is created inside the `dataloader_creation` span and used in the `prediction` span. This is correct Python — the `with mlflow.start_span(...)` context manager closes the span when the block exits but does not affect variable scope. `test_loader` remains accessible in the outer function scope for the subsequent span.
- `mlflow.start_span` does not require an active `mlflow.start_run` context in MLflow 3.x — traces are logged directly against the experiment without creating a run entry. This keeps API inference traces separate from training runs in the Experiments tab.
- The `model/unittests/test_api/` directory must **not** have an `__init__.py`. Importing `api.py` triggers `prg_torch_model.py` module-level code (model instantiation from HuggingFace weights), which blocks test discovery. Without `__init__.py`, the directory is not auto-discovered and tests must be run directly: `uv run python -m unittest test_api.test_classify_image_spans`. Follow the same no-`__init__.py` pattern as `unittests/utilities/`.
- The smoke test mocks `model.load()` so no `.pt` file is required on disk. The smoke test patches the `load` method on the model instance before making the request, allowing the full span structure to be verified without real model weights.
- `model/api.py` imports `prg_torch_model.model_dict` and `prg_torch_model.torch_transforms` at line 10. This triggers `prg_torch_model.py` module-level code (model instantiation) at import time. Both files call `mlflow.set_tracking_uri` to the same value, which is idempotent and safe.
- The `**classification_result` spread in `root_span.set_outputs({"status_code": 200, **classification_result})` will include all keys from the response dict. MLflow span output values must be JSON-serialisable primitives (str, int, float, bool, None). The `classification_result` dict contains `datetime` (str), `model_id` (str), `category_name` (str), `probability` (float), `category_id` (int), `ndims` (int), `torch_transform_error` (bool/None), and file path strings — all safe. If a new key with a non-serialisable value is ever added to the response dict in future, it will cause a span output serialisation error at runtime.
