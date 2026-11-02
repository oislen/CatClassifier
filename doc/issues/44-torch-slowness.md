# Issue #44 — Torch Slowness

## Overview

Investigate and resolve slowness when training and classifying images with PyTorch, with a specific focus on the custom dataset generation pipeline. The issue was identified through timing data captured in the MLflow UI showing unexpectedly high data processing times for both the image classification API (`classify_image_torch.py`) and the model training pipeline (`prg_torch_model.py`).

The root cause is that all image loading and transformation work is performed upfront and sequentially in a single process before the `DataLoader` is constructed. The `TorchLoadImages.load_images` call on 20,000 training images — each involving `Image.open`, transform application (resize + ToTensor + Normalize), and `np.asarray` for the raw pixel array — runs entirely in the main thread with no parallelism. The resulting tensors are then materialised into a Pandas DataFrame and held in memory before any training begins.

**Issue:** [#44 - Torch Slowness](https://github.com/oislen/CatClassifier/issues/44)  
**Milestone:** Cat Classifier Version 1  
**Label:** enhancement

---

## Root Cause Analysis

### Current data pipeline

```
os.listdir(train_fdir)
    → TorchLoadImages.load_images(filepaths)      # sequential, main thread
        → for filepath in filepaths:
            Image.open(filepath)                   # disk I/O
            torch_transforms(image)                # resize + ToTensor + Normalize (CPU)
            np.asarray(image)                      # full pixel array stored in memory
            image.close()
    → pd.DataFrame.from_records(records)           # all 20,000 tensors in memory
    → CustomDataset(df)                            # wraps pre-loaded tensors
    → DataLoader(dataset, num_workers=0, ...)      # serves pre-loaded tensors
```

There are four compounding bottlenecks:

1. **Sequential loading** — `n_workers=None` in `TorchLoadImages` forces single-threaded loading. Each image is loaded and transformed one after the other in the main process. The `multiprocessing.Pool` path exists but is never activated (`n_workers=None` in both `prg_torch_model.py` and `classify_image_torch.py`).

2. **Redundant raw pixel copy** — `load_image` stores `np.asarray(image)` (the full raw pixel array) in the `images` column of the DataFrame alongside the already-transformed `image_tensors`. For 20,000 RGB images this doubles memory consumption unnecessarily during training.

3. **Pre-loading all images into memory** — the entire dataset is loaded and transformed before training begins. PyTorch's standard pattern defers per-image I/O and transforms to `Dataset.__getitem__`, which enables `DataLoader` workers to parallelise loading across batches. The current approach eliminates that opportunity. **This bottleneck is not addressed by this plan** — a lazy-loading refactor is out of scope and deferred to a follow-up ticket (see Notes).

4. **`num_workers=0` in `DataLoader`** — `cons.num_workers` defaults to `0` via `PARAM_NUM_WORKERS`, meaning no background prefetching occurs during model forward/backward passes. Since `CustomDataset.__getitem__` returns pre-loaded in-memory tensors, the per-batch overhead is low and the benefit of `DataLoader` workers is limited on CPU; it is more significant on CUDA where workers overlap host-to-device transfer with the forward pass. Addressed by the execution script changes in Section 3.

### Evidence from screenshots

The MLflow screenshots referenced in the issue show:
- **Flask API** (`classify_image_torch.py`): elevated `DataPrep.TrainDataLoad` time for a single inference image.
- **Training pipeline** (`prg_torch_model.py`): significant `DataPrep.TrainDataLoad` time relative to the total training time, indicating data loading — not model computation — is the bottleneck.

### Dataset sizes

| Split | Image count | Loading strategy |
|---|---|---|
| Training | 20,000 (sampled from 37,000) | Parallel — large enough to justify worker spawn overhead |
| Test | 12,500 | Parallel — large enough to justify worker spawn overhead |
| API inference | 1 | Sequential — process spawn overhead exceeds any benefit |

---

## Scope

This plan covers four areas:

1. Enable multiprocessing in `TorchLoadImages` for both training and test set loading
2. Remove the redundant raw pixel array copy from `load_image`
3. Add `PARAM_NUM_WORKERS` to execution scripts so parallel loading is enabled without changing the safe `0` default in `cons.py`
4. Add `persistent_workers` to `DataLoader` calls in `prg_torch_model.py` to avoid repeated worker respawning across epochs when `num_workers > 0`

> **Note:** `persistent_workers` and `DataLoader` `num_workers > 0` were reverted after testing showed increased inter-epoch latency. With pre-loaded in-memory tensors, `CustomDataset.__getitem__` is near-zero cost and `DataLoader` workers add IPC overhead per batch rather than saving time. All three `DataLoader` calls use `num_workers=0` — `PARAM_NUM_WORKERS` activates parallelism in `TorchLoadImages` only (the upfront loading phase), not in the batch-serving phase.

A full refactor to lazy-loading `Dataset.__getitem__` is noted as a follow-up option in the Notes section; it is out of scope for this ticket because it requires a structural redesign of `CustomDataset` and `TorchLoadImages`.

---

## Prerequisites

No new dependencies are required. All changes use the Python standard library and existing project packages.

---

## Implementation Plan

### 1. Enable multiprocessing in `TorchLoadImages`

**Goal:** Activate the existing `multiprocessing.Pool` path in `TorchLoadImages` when loading large batches of images, reducing the sequential loading bottleneck for both the training and test set pipelines.

**Changes required:**

#### `model/arch/load_image_v2.py`

The `multi_process` method creates a `Pool` correctly but the activation condition (`n_workers=None`) was never triggered in either caller. Both `multiprocessing.Pool` and `concurrent.futures.ProcessPoolExecutor` work with bound methods when the class is defined in an importable module (as `TorchLoadImages` is in `load_image_v2.py`) — the `spawn` start method used on Windows resolves the class by re-importing the module in each worker process. Replace `multiprocessing.Pool` with `concurrent.futures.ProcessPoolExecutor` for a cleaner context-manager interface and consistent exception propagation:

```python
import os
import re
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from PIL import Image


def load_image_v2(image_fpaths):
    """ """
    images = []
    for image_fpath in image_fpaths:
        temp = Image.open(image_fpath)
        keep = deepcopy(temp)
        images.append(keep)
        temp.close()
    return pd.Series(images)


class TorchLoadImages:
    def __init__(self, torch_transforms, n_workers=None):
        self.torch_transforms = torch_transforms
        self.n_workers = n_workers

    def load_image(self, filepath):
        """ """
        # determine the filename and source
        file_name = os.path.basename(filepath)
        # determine label from image file path
        if ("cat" in file_name) or ("dog" in file_name):
            file_source = (
                "kaggle"
                if len(re.findall(pattern=r"^(cat|dog)(\.\d+\.jpg)$", string=file_name))
                > 0
                else "webscraper"
            )
            label_name = file_name.split(".")[0]
            label = label_name == "dog"
            label_tensor = torch.tensor(label, dtype=torch.int64)
        else:
            file_source = "kaggle"
            label_name = np.nan
            label = np.nan
            label_tensor = torch.tensor(label)
        # load image file and apply torch transforms
        image = Image.open(filepath)
        torch_transform_error = None
        try:
            image_tensor = self.torch_transforms(image)
        except Exception as e:
            image_tensor = None
            torch_transform_error = str(e)
        n_dims = len(np.asarray(image).shape)
        image.close()
        # create an output record — raw pixel array is not stored to reduce memory usage
        record = {
            "filepaths": filepath,
            "filenames": file_name,
            "source": file_source,
            "categoryname": label_name,
            "category": label,
            "ndims": n_dims,
            "category_tensors": label_tensor,
            "image_tensors": image_tensor,
            "torch_transform_error": torch_transform_error,
        }
        return record

    def load_images(self, filepaths):
        """ """
        if self.n_workers is None or self.n_workers <= 1:
            records = [self.load_image(filepath) for filepath in filepaths]
        else:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                records = list(executor.map(self.load_image, filepaths))
        return records
```

Key changes:
- Replace `multiprocessing.Pool` with `concurrent.futures.ProcessPoolExecutor` for a context-manager interface and cleaner exception propagation.
- Remove `multi_process` as a separate method — the parallelism is now inlined in `load_images`.
- Remove the `images` column from the output record (see Section 2).
- Guard `n_workers <= 1` so a value of `1` also falls through to the sequential path (avoids subprocess overhead for single-worker usage).

> **Windows note:** both `multiprocessing.Pool` and `ProcessPoolExecutor` use the `spawn` start method on Windows. Worker processes resolve `TorchLoadImages` by importing `load_image_v2` — since the class is in an importable module, pickling of the bound method succeeds. The `if __name__ == "__main__":` guard in `prg_torch_model.py` already protects the main script entry point. Multiprocessing calls from within the main block are safe.

---

#### `model/prg_torch_model.py`

Pass `n_workers` from `cons` to `TorchLoadImages` in both the training data loading block and the testset prediction block. With `PARAM_NUM_WORKERS` set in the execution scripts (Section 3), `cons.num_workers` will be a positive integer and the parallel path will activate. The `api.py` and `classify_image_torch.py` callers are unaffected — they continue to pass `n_workers=None` explicitly and `cons.num_workers` is only used for the `DataLoader`, not for `TorchLoadImages`, in those scripts.

```python
# data_loading span — training pipeline
torch_load_images = TorchLoadImages(
    torch_transforms=torch_transforms, n_workers=cons.num_workers if cons.num_workers > 0 else None
)
```

```python
# testset_prediction span — test set pipeline (12,500 images)
torch_load_images = TorchLoadImages(
    torch_transforms=torch_transforms, n_workers=cons.num_workers if cons.num_workers > 0 else None
)
```

The `n_workers=cons.num_workers if cons.num_workers > 0 else None` expression converts a `PARAM_NUM_WORKERS=0` value back to `None`, which keeps the sequential path when workers are explicitly disabled (e.g. in CI). When `PARAM_NUM_WORKERS` is not set, `cons.num_workers` remains `0` (the existing safe default is unchanged), so both blocks fall back to sequential loading unless the environment variable is explicitly set.

---

### 2. Remove redundant raw pixel array storage

**Goal:** Eliminate the `images` column from the loaded DataFrame to halve memory consumption during data loading. The `images` column stores an `np.asarray(image)` copy of every raw pixel array alongside the already-transformed `image_tensors`. For 20,000 training images at 128×128 pixels this is approximately 983 MB of redundant storage. It is only used by `plot_image` (for the diagnostic plot) and nowhere else during training or inference.

**Changes required:**

#### `model/arch/load_image_v2.py`

The `images` key is already removed from `load_image` in the updated code shown in Section 1. The `ndims` value is computed inline from a temporary array without storing the full array:

```python
n_dims = len(np.asarray(image).shape)
image.close()
```

Note: `np.asarray(image)` must be called before `image.close()` — calling it on a closed PIL image raises `Operation on closed image`.

#### `model/prg_torch_model.py`

The `plot_image` call in the `data_loading` span uses `df["images"].values[1]`. Replace it with a direct `Image.open` from the filepath already stored in `df["filepaths"]`. Note that `image_filepaths` is deleted at line 143 (`del image_filepaths`) before the `plot_image` call at line 148, so `image_filepaths[1]` cannot be used — `df["filepaths"].values[1]` must be used instead:

```python
logging.info("Plot sample image...")
sample_image_fpath = df["filepaths"].values[1]
sample_image_array = np.asarray(Image.open(sample_image_fpath))
plot_image(
    sample_image_array,
    output_fpath=cons.torch_random_image_fpath,
    show_plot=False,
)
time_logger.log_time(parent_key="Plots", sub_key="SampleImage")
```

Add `from PIL import Image` to the imports in `prg_torch_model.py` if not already present.

---

### 3. Update execution scripts to set `PARAM_NUM_WORKERS`

**Goal:** Enable parallel loading in both training and test set pipelines by setting `PARAM_NUM_WORKERS` in the execution scripts. The `cons.py` default of `0` is left unchanged — parallel loading is opt-in via the environment variable, keeping the behaviour safe across all environments (CI, Docker, API) that do not set the variable.

**Changes required:**

#### `model/exeTorchModel.cmd`

Add `SET PARAM_NUM_WORKERS` before the `uv run` invocations:

```cmd
call SET MLFLOW_TRACKING_URI=http://127.0.0.1:5000
call SET PARAM_CHECK_GPU=True
call SET PARAM_NUM_WORKERS=4
call uv run python prg_torch_model.py --run_model_training --run_testset_prediction --model_id alexnet_8_pretrained --device_type cuda
:: call uv run python prg_torch_model.py --run_model_training --run_testset_prediction --model_id vgg_16_pretrained --device_type cuda
:: call uv run python prg_torch_model.py --run_model_training --run_testset_prediction --model_id resnet_50_pretrained --device_type cuda
```

#### `model/exeTorchModel.sh`

Add `export PARAM_NUM_WORKERS` before the `uv run` invocations:

```sh
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export PARAM_CHECK_GPU=True
export PARAM_NUM_WORKERS=4
#uv run python prg_torch_model.py --run_model_training --run_testset_prediction --model_id alexnet_8_pretrained
uv run python prg_torch_model.py --run_model_training --run_testset_prediction --model_id vgg_16_pretrained
#uv run python prg_torch_model.py --run_model_training --run_testset_prediction --model_id resnet_50_pretrained
```

> Tune `PARAM_NUM_WORKERS` to match the environment. A value of `4` is a reasonable starting point for a multi-core development machine. Set to `0` (or leave unset) in CI or Docker environments with limited CPU resources to keep the safe sequential default.

---

## File Change Summary

| File | Change Type | Description |
|---|---|---|
| `model/arch/load_image_v2.py` | Modify | Replace `multiprocessing.Pool` with `concurrent.futures.ProcessPoolExecutor`; remove `multi_process` method; remove `images` column from `load_image` output record; compute `ndims` inline without storing raw pixel array |
| `model/prg_torch_model.py` | Modify | Pass `n_workers` from `cons` to `TorchLoadImages` in both training and testset prediction blocks; replace `df["images"].values[1]` with `Image.open(df["filepaths"].values[1])` for sample plot; add `from PIL import Image` import; all three `DataLoader` calls use `num_workers=0` explicitly |
| `model/exeTorchModel.cmd` | Modify | Add `SET PARAM_NUM_WORKERS=4` |
| `model/exeTorchModel.sh` | Modify | Add `export PARAM_NUM_WORKERS=4` |

---

## Testing

### Manual performance benchmarking

Run the training pipeline before and after the changes and compare the `DataPrep.TrainDataLoad` timing values logged in the MLflow UI:

```cmd
:: from CatClassifier/model/
call SET MLFLOW_TRACKING_URI=http://127.0.0.1:5000
call SET PARAM_NUM_WORKERS=4
call uv run python prg_torch_model.py --run_model_training --model_id alexnet_8_pretrained --device_type cpu
```

Expected improvement: `DataPrep.TrainDataLoad` time should decrease roughly in proportion to the number of workers used (e.g. 4 workers → ~4x reduction for I/O-bound loading), subject to OS scheduling and disk throughput limits. The test set load time (`DataPrep.TrainDataLoad` in the `testset_prediction` span) should see a similar reduction across the 12,500 test images.

### Unit tests

#### New test file: `model/unittests/test_arch/test_TorchLoadImages.py`

> **Fixture requirement:** `TorchLoadImages.load_image` calls `Image.open(filepath)` and requires a real image file on disk. Tests should create a minimal synthetic JPEG fixture using `PIL.Image.new("RGB", (10, 10)).save(tmp_path)` in `setUpClass` and remove it in `tearDownClass`. The parallel tests (`test_load_images_parallel`, `test_load_images_parallel_results_match_sequential`) require at least two fixture files so the pool has work to distribute. Tests must be wrapped under `if __name__ == "__main__":` or use `unittest.main()` to avoid multiprocessing re-entry on Windows.

| Test name | What it verifies |
|---|---|
| `test_load_image_returns_dict` | `load_image` returns a `dict` |
| `test_load_image_record_has_no_images_key` | The record does not contain an `images` key (regression guard for the memory reduction) |
| `test_load_image_record_has_ndims_key` | The record contains `ndims` |
| `test_load_image_ndims_is_int` | `ndims` is an `int` |
| `test_load_images_sequential` | `load_images` with `n_workers=None` returns the correct number of records |
| `test_load_images_parallel` | `load_images` with `n_workers=2` returns the same number of records as the sequential path |
| `test_load_images_parallel_results_match_sequential` | Record contents produced by the parallel path match the sequential path for the same input filepaths (validates correctness of `ProcessPoolExecutor` usage) |
| `test_load_images_n_workers_1_uses_sequential_path` | `n_workers=1` falls through to the sequential path without spawning a subprocess pool (verifies the `<= 1` guard) |

#### New test file: `model/unittests/utilities/test_cons_num_workers.py`

> **Import note:** `cons.num_workers` is assigned at module import time from `os.environ`. Tests that override `PARAM_NUM_WORKERS` must set the environment variable **before** importing `cons`, then use `importlib.reload(cons)` to force re-evaluation, and restore the original value in `tearDown`. Alternatively, patch `os.environ` with `unittest.mock.patch.dict` before triggering the reload.

| Test name | What it verifies |
|---|---|
| `test_num_workers_is_int` | `cons.num_workers` is an `int` |
| `test_num_workers_default_is_zero` | `cons.num_workers` equals `0` when `PARAM_NUM_WORKERS` is not set (confirms safe default is unchanged) |
| `test_num_workers_env_override` | Setting `PARAM_NUM_WORKERS=4` in the environment and reloading `cons` causes `cons.num_workers` to equal `4` (tests the environment variable override path) |

---

## Notes

- **`api.py` and `classify_image_torch.py` are unaffected:** both pass `n_workers=None` to `TorchLoadImages` (single-image inference — no benefit from worker spawn overhead) and neither sets `PARAM_NUM_WORKERS`, so `cons.num_workers` remains `0` and their `DataLoader` workers are unchanged. No modifications to those files are required.
- **`DataLoader` `num_workers` vs `TorchLoadImages` `n_workers`:** these are separate concerns. `TorchLoadImages.n_workers` controls parallelism during the upfront dataset generation phase (loading all images before training begins) and benefits both CPU and CUDA training. `DataLoader` `num_workers` controls background batch prefetching during training; its benefit depends on the device — on CUDA it overlaps host-to-device transfer with the forward pass, but with pre-loaded in-memory tensors on CPU the workers add process-spawn overhead with minimal throughput gain. Both parameters are currently controlled by the same `PARAM_NUM_WORKERS` variable; this is a reasonable starting point, but the two are independently tunable if fine-grained control is needed.
- **`persistent_workers` — reverted after testing:** setting `persistent_workers=True` with `num_workers > 0` caused increased inter-epoch latency in practice. With pre-loaded in-memory tensors, `CustomDataset.__getitem__` is near-zero cost — each batch fetch is just an array lookup. Worker IPC overhead on Windows `spawn` exceeds the work being parallelised, making `num_workers > 0` actively slower for batch serving. All `DataLoader` calls remain at `num_workers=0`. `PARAM_NUM_WORKERS` applies only to `TorchLoadImages` (upfront loading), which has substantial I/O and CPU transform work per image and genuinely benefits from parallelism. If a lazy-loading refactor is implemented in future (deferred — see below), `DataLoader` `num_workers > 0` would become beneficial and `persistent_workers=True` should be reconsidered at that point.
- **Memory vs speed trade-off:** removing the `images` column reduces peak memory usage during loading by approximately `20000 × 128 × 128 × 3 × 1` bytes (raw uint8 pixels) ≈ 983 MB. The transformed `image_tensors` (float32) occupy a similar amount. Removing `images` makes the peak memory roughly half of what it was during the upfront loading phase.
- **Full lazy-loading refactor (future work):** the most thorough fix is to refactor `CustomDataset.__getitem__` to load and transform images on demand (storing only filepaths in the dataset, not pre-loaded tensors). This would eliminate the upfront loading cost entirely and allow `DataLoader` workers to parallelise all I/O and CPU transforms across batches. This is a more significant structural change and is deferred to a follow-up ticket, as it requires changes to `CustomDataset`, `TorchLoadImages`, and the DataFrame construction logic in `prg_torch_model.py`, `classify_image_torch.py`, and potentially `api.py`.
- **Windows multiprocessing and `spawn` start method:** on Windows, both `multiprocessing.Pool` and `ProcessPoolExecutor` use the `spawn` start method. Each worker process re-imports `load_image_v2` to resolve `TorchLoadImages` — this works because the class is in an importable module. The `if __name__ == "__main__":` guard in `prg_torch_model.py` prevents re-execution of training code in worker processes. No additional guard is needed in `load_image_v2.py` because it does not have a `__main__` block.
- **`num_workers` in Docker:** the `compose.yaml` container has `PARAM_NUM_WORKERS` unset, so it inherits the safe `0` default from `cons.py`. To enable parallel loading inside the container, add `PARAM_NUM_WORKERS: "4"` to the `environment:` block of the relevant service in `compose.yaml`. The value should be set to match the container's CPU allocation, not the host's core count.
- **Profiling recommendation:** before and after the change, use `cProfile` or the `TimeIt` logger already present in `prg_torch_model.py` to quantify the improvement. MLflow's `training_time_seconds` metric and the per-span durations in the Traces tab provide a built-in benchmark without any additional tooling.
- **Per-worker `torch`/`torchvision` import overhead:** each `ProcessPoolExecutor` worker re-imports `torch` and `torchvision` on startup. On a cold start this adds a one-time overhead per worker process (typically 1–3 seconds). This cost is amortised across the full 20,000-image or 12,500-image load, so it is negligible at those scales but would outweigh the benefit for small batches (tens of images). The `n_workers <= 1` guard in `load_images` already prevents this overhead for single-image callers.
- **`logging.info` in the inner training batch loop (out of scope — follow-up):** `fit.py` calls `logging.info` on every batch step (line 103–105). At `batch_size=64` and 20,000 training images this produces 312 log lines per epoch and 3,120 over 10 epochs. On CUDA each `loss.item()` call forces a device synchronisation, which can meaningfully slow the training loop. On CPU the cost is purely I/O to the log handler. Reducing this to one `logging.info` per epoch (or removing it entirely and relying on the MLflow per-epoch metrics already logged at lines 114–120) would clean up both the log output and the training loop latency. This is out of scope for this ticket but is worth tracking as a follow-up.
