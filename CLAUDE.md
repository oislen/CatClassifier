# CatClassifier

## Project Overview

CNN-based image classifier that distinguishes cats from dogs. Three pretrained PyTorch models are fine-tuned (AlexNet8, VGG16, ResNet50) on data from the Kaggle dogs-vs-cats competition and freeimages.com (via web scraper). A Flask REST API serves predictions; the app is containerised and deployed to Docker Hub / AWS EC2 spot instances.

## Tech Stack

- **Python 3.13.11** — managed with [uv](https://docs.astral.sh/uv/)
- **PyTorch / TorchVision** (CUDA 12.9 wheel index) — model training & inference
- **Flask** — REST API
- **BeautifulSoup4 / Kaggle SDK** — data collection
- **Ruff** — linting & formatting
- **Docker** — containerisation; image published to Docker Hub as `oislen/cat-classifier`

## Repository Layout

```
model/
  arch/              # image preprocessing helpers (crop, pad, resize, greyscale, load)
  torch_models/      # model definitions and training helpers (fit, validate, predict, checkpoints, early stopping)
  utilities/         # shared utilities (e.g. TimeIt)
  unittests/         # unit tests for model utilities
  cons.py            # project-wide constants and path configuration
  api.py             # Flask REST API
  prg_torch_model.py # CLI training / prediction script
  exeTorchModel.cmd  # Windows shortcut to run training
  exeApi.cmd         # Windows shortcut to start the API locally
webscrapers/
  arch/              # scraper implementations
  unittests/         # unit tests for scraper utilities
  prg_scrape_imgs.py # CLI data-download / model-download script
config/
  uv/                # uv lockfile / environment config
  conda/             # legacy conda environment spec
data/                # training images, checkpoints, serialised models (git-ignored)
report/              # generated plots and analysis notebooks
doc/                 # architecture diagram, Postman collection
```

## Environment Setup

Dependencies are managed with **uv**. Python 3.13.11 is required (see `.python-version`).

```bash
uv sync
```

PyTorch is installed from the CUDA 12.9 index automatically via `pyproject.toml`.

### VS Code Dev Container

The `.devcontainer/devcontainer.json` builds the project `Dockerfile` and mounts:
- `.creds/` → Kaggle API credentials
- `report/` → generated output

The container starts the Flask API automatically (`postStartCommand`) and forwards port 5000.

## Running Commands

All Python commands should be prefixed with `uv run` (or `uv run --active` inside an active venv).

### Training a model (Windows)

```cmd
cd model
call exeTorchModel.cmd
```

Equivalent long-form:

```bash
cd model
uv run python prg_torch_model.py \
  --run_model_training \
  --run_testset_prediction \
  --model_id alexnet_8_pretrained \
  --device_type cuda
```

Available `--model_id` values: `alexnet_8_pretrained`, `vgg_16_pretrained`, `resnet_50_pretrained`.

### Flask API (local)

```cmd
cd model
call exeApi.cmd
# or:
uv run flask --app api.py run --debug --host 0.0.0.0 --port 5000
```

POST an image to `http://localhost:5000/catclassifier` with fields:
- `image` — image file
- `model_id` (optional) — one of the model IDs above
- `device_type` (optional) — `cpu` or `cuda`

A Postman collection is in `doc/CatClassifier.postman_collection.json`.

### Docker (production)

```bash
# Flask API
docker run --name cc --gpus all --publish 5000:5000 \
  --env PARAM_CHECK_GPU=True oislen/cat-classifier:latest

# Jupyter Lab
docker run --name cc --shm-size=512m --entrypoint uv \
  --publish 8888:8888 --gpus all --env PARAM_CHECK_GPU=True \
  -it oislen/cat-classifier:latest \
  run jupyter lab --ip=0.0.0.0 --allow-root \
  --IdentityProvider.token='' --ServerApp.password=''
# Then open http://127.0.0.1:8888/lab
```

## Running Tests

Tests use Python's built-in `unittest` via the Makefile:

```bash
# Run all tests (model + webscrapers)
make test

# Run a single test module
make test T=model/unittests/utilities/test_TimeIt.py

# Run a single test case
make test T=model.unittests.utilities.test_TimeIt.Test_TimeIt
```

CI sets `GITHUB_ACTIONS_UNITTEST_FLAG=1`; some tests branch on this variable.

## Linting & Formatting

```cmd
call exeLinters.cmd
```

This runs in order:
1. `ruff check --fix` — lint with auto-fix
2. `ruff format` — format
3. `zizmor . --fix=all` — GitHub Actions workflow security audit

Ruff configuration is in `pyproject.toml` (`[tool.ruff]`). Target is Python 3.12 syntax, line length 88. `model/arch/keras_model.py` is excluded.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PARAM_CHECK_GPU` | `False` | Set `True` to enable CUDA if available |
| `PARAM_NUM_WORKERS` | `0` | DataLoader worker processes |
| `FLASK_DEBUG` | `""` | Set `1`/`true`/`yes` to enable Flask debug mode |
| `TORCH_HOME` | — | Override PyTorch model cache directory |

Kaggle API credentials are required to download training data. Place `kaggle.json` in `.creds/` (mounted into the container at `/home/ubuntu/CatClassifier/.creds`).

## CI/CD (GitHub Actions)

### Pull Requests (`dev-pull-requests.yml`)

Runs on PRs to `dev` and `main`:
- Semantic PR title validation
- Ruff lint check
- SonarQube scan (self-hosted runner, skipped if server offline)
- Trivy filesystem vulnerability scan (fails on HIGH/CRITICAL)
- Python unit tests (Python 3.13.11)
- Zizmor CI/CD security check

### Main Push (`deploy-main-push.yml`)

Runs on push to `main`; same quality gates plus:
- **release-please** — automated changelog and version bump (requires `trivy-fs-scan`, `python-unittests`, `ruff-linter` to pass)
- **dockerhub-image-push** — builds and pushes Docker image to `oislen/cat-classifier:<tag>` and `:latest` (only when a release is created; requires `KAGGLE_USERNAME` / `KAGGLE_KEY` secrets to download model weights)
- **trivy-image-scan** — scans the published Docker image

Required GitHub secrets: `SONAR_PROJECT_KEY`, `SONAR_TOKEN`, `SONAR_HOST_URL`, `KAGGLE_USERNAME`, `KAGGLE_KEY`, `DOCKERHUB_TOKEN`.  
Required GitHub variables: `DOCKERHUB_USERNAME`, `DOCKERHUB_REPO`.

## Key Constants (`model/cons.py`)

| Constant | Value |
|---|---|
| `IMAGE_SIZE` | 128 × 128 |
| `IMAGE_CHANNELS` | 3 |
| `batch_size` | 64 |
| `learning_rate` | 0.001 |
| `min_epochs` / `max_epochs` | 3 / 10 |
| `category_mapper` | `{0: "cat", 1: "dog"}` |

Training uses SGD with cross-entropy loss, learning-rate reduction on plateau, and early stopping.
