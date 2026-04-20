# VisionDX — Multi-Disease Retinal Diagnosis

## Overview

VisionDX is a full-stack prototype for retinal disease detection using deep learning. It includes:

- A PyTorch-based model training and evaluation pipeline in `src/`
- A FastAPI backend in `Backend/src/` for image upload and inference
- A React + Vite frontend in `frontend/` for interactive diagnosis
- Pretrained model weights in `models/resnet50_retinal.pth`

The system predicts one of several retinal conditions from fundus images and returns both the predicted label and confidence scores.

## Key Features

- Image upload endpoint at `/predict`
- ResNet-50 based retinal classifier with 5 disease outputs
- Dark-mode React UI with image preview and results display
- Training and evaluation support through `main.py` and `src/` utilities

## Repository Structure

- `Backend/` — FastAPI backend source, model serving, and API logic
  - `Backend/src/app.py` — API server
  - `Backend/src/model.py` — Retina network definition
  - `Backend/src/predict_utils.py` — image preprocessing and prediction helpers
- `frontend/` — React UI built with Vite
  - `frontend/src/App.jsx` — main app interface
  - `frontend/package.json` — frontend dependencies and scripts
- `src/` — model training / data pipeline utilities
  - `src/dataloader_setup.py` — data loading and train/validation splitting
  - `src/train.py` — training loop and checkpoint saving
  - `src/evaluate.py` — evaluation metrics and confusion matrix plotting
  - `src/model.py` — ResNet model used for training
- `data/` — dataset metadata and image folders
  - `data/train.csv`, `data/train/`, `data/val_labels.csv`, `data/val/`
- `models/` — saved model weights and class labels
  - `models/resnet50_retinal.pth`
  - `models/class_names.txt`

## Requirements
 
### Python

- Python 3.10+ recommended
- Backend dependencies in `Backend/requirements.txt`

### Node

- Node.js 18+ recommended for running the React frontend

## Setup

### 1. Install backend dependencies

```bash
cd Backend
python -m pip install -r requirements.txt
```

### 2. Install frontend dependencies

```bash
cd ../frontend
npm install
```

### 3. Prepare model and data

- Confirm `models/resnet50_retinal.pth` exists
- Confirm `models/class_names.txt` contains one label per line
- Ensure `data/train.csv` and `data/train/` are populated
- Optionally provide `data/val_labels.csv` and `data/val/` for validation

## Running Locally

### Start the backend API

From the `Backend/` folder:

```bash
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

### Start the frontend app

From the `frontend/` folder:

```bash
npm run dev
```

Then open the Vite dev server URL shown in the terminal (typically `http://localhost:5173`).

## Training a New Model

The root `main.py` runs the training and evaluation pipeline:

```bash
python main.py
```

It uses:

- `src/dataloader_setup.py` for dataset loading and splitting
- `src/train.py` for training and best-model checkpointing
- `src/evaluate.py` for accuracy and classification reporting

The trained weights are saved to `models/resnet50_retinal.pth`.

## API Usage

Send a `POST` request to `/predict` with a form file field named `file`:

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@/path/to/image.jpg"
```

Response format:

```json
{
  "predicted_class": "Diabetic Retinopathy",
  "confidences": {
    "Normal": 0.03,
    "Diabetic Retinopathy": 0.92,
    "Glaucoma": 0.02,
    "Cataract": 0.01,
    "AMD": 0.02
  }
}
```

## Notes

- The backend uses CPU or CUDA automatically via PyTorch.
- The frontend defaults to `http://localhost:8000`; set `VITE_API_URL` if your API runs elsewhere.
- The current model uses a ResNet-50 backbone and expects 224×224 fundus images.

## Environment Configuration

The frontend can be configured using `frontend/.env` or `frontend/.env.local`.
Use the example file `frontend/.env.example` to get started.

```env
VITE_API_URL=http://localhost:8000
```

If deploying the frontend separately, set `VITE_API_URL` to your hosted backend URL.

## Contributing

Contributions are welcome! A good way to start is:

1. Fork the repository.
2. Create a new branch for your feature or fix.
3. Add or improve documentation, fix issues in `Backend/src/`, `src/`, or `frontend/src/`.
4. Test locally with `uvicorn src.app:app --reload --host 0.0.0.0 --port 8000` and `npm run dev`.
5. Open a pull request with a clear description of your changes.

## Enhancements

Possible future improvements:

- Add model explainability visualizations
- Improve dataset preprocessing and augmentation
- Add authentication and user management
- Provide a production-ready deployment pipeline
