# AI Fake Image Detector

This project detects whether a face image is likely real or AI-generated. It uses a PyTorch ResNet-18 binary classifier, local image metadata CSVs, command-line inference, and a Flask web interface.

## Current Status

- Model training and inference work locally.
- The local dataset cache contains all 6,557 source images: 2,790 real and 3,767 fake.
- `best_model.pth` and `best_model_finetuned.pth` currently contain identical weights. Keep them for traceability, but do not claim they are distinct models.
- `best_model_full.pth` is the preferred checkpoint trained on the complete local CSV.

## Project Structure

```text
project/
  app.py                         Flask web app
  model.py                       ResNet-18 model definition
  dataset.py                     PyTorch Dataset and DataLoader creation
  train.py                       Training and evaluation script
  predict.py                     CLI and reusable prediction helpers
  user_predict.py                Interactive terminal prediction script
  validate_project.py            Local project health/status report
  download_dataset.py            Basic dataset downloader
  download_dataset_resumable.py  Resumable downloader for missing images
  prepare_external_dataset.py    Builds CSVs from external real/fake folders
  templates/index.html           Web upload UI
  static/styles.css              Web UI styling
  FINAL_DATASET.csv              Source URL dataset
  dataset_all_local.csv          Best current local metadata CSV
  best_model.pth                 Saved model checkpoint
  best_model_finetuned.pth       App default checkpoint
  best_model_full.pth            Preferred full-dataset checkpoint
```

## Setup

Use a virtual environment. Python 3.10+ is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Check Project Health

```bash
python validate_project.py
```

For JSON output:

```bash
python validate_project.py --json
```

Run smoke tests:

```bash
python -m unittest discover -s tests
```

## Complete Or Resume Dataset Download

Use the resumable downloader if files are deleted or you need to rebuild the local cache without restarting from zero. Dead Multiavatar JPG URLs automatically fall back to DiceBear-generated PNG data.

```bash
python download_dataset_resumable.py --threads 8
```

The canonical local training CSV should be:

```text
dataset_all_local.csv
```

## Train

Train on the current local dataset:

```bash
python train.py --csv_path dataset_all_local.csv --epochs 3 --batch_size 32 --lr 0.0001 --fine_tune false --initial_weights best_model.pth --checkpoint_path best_model_full.pth --summary_path training_summary_full.json
```

For a faster CPU smoke test:

```bash
python train.py --csv_path dataset_local.csv --epochs 1 --batch_size 16 --fine_tune false
```

## Predict From Terminal

```bash
python predict.py --image_path data\real\1.jpg --model_path best_model_full.pth
```

Interactive mode:

```bash
python user_predict.py
```

## Run Web App

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

Health endpoint:

```text
http://127.0.0.1:5000/health
```

The app automatically prefers `best_model_full.pth`, then `best_model_finetuned.pth`, then `best_model.pth`. To force a specific model checkpoint:

```bash
set FAKE_IMAGE_MODEL_PATH=best_model.pth
python app.py
```

## Metrics From Latest Saved Summaries

Base training run:

- Validation accuracy: 96.39%
- Validation precision: 99.59%
- Validation recall: 92.93%
- Validation F1: 96.14%

Fine-tuning summary:

- Validation accuracy: 99.26%
- Validation precision: 100.00%
- Validation recall: 98.47%
- Validation F1: 99.23%

Full-dataset checkpoint:

- Dataset: `dataset_all_local.csv`
- Epochs: 3
- Validation accuracy: 98.86%
- Validation precision: 99.33%
- Validation recall: 98.67%
- Validation F1: 99.00%

Do not oversell these numbers. They are validation-split metrics from local data, not proof of real-world production performance. For a stronger project, add an independent external test set and report results separately.

## Practical Next Steps

1. Add an independent external test CSV with images from sources not used in training.
2. Report validation and external-test metrics separately.
3. Keep large model/data files out of git if you publish this repository.

## Model:
FakeImageClassifier — ResNet-18 backbone (ImageNet pretrained by default).

## Framework:
 PyTorch.