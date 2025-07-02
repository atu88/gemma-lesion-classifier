# Skin Lesion Scripts

This repository contains multiple Python scripts designed to support preprocessing, training, inference, and evaluation for skin lesion image classification tasks. These scripts cover data cleaning, conversion, splitting, model fine-tuning, inference, and performance evaluation.

---

## Scripts Overview

### 1. Image Quality Filtering and Deduplication (`duplicates_removal.py`)
- Filters and scores skin lesion images based on blur, entropy, and edge density.
- Detects images with dark borders and deprioritizes them.
- Retains the best-quality image per lesion and moves duplicates/artifact-heavy images to a separate folder.
- Outputs cleaned metadata and a list of rejected images.

### 2. Metadata CSV to JSONL Conversion (`jsonl_convert.py`)
- Converts CSV metadata containing image filenames and labels to JSON Lines format (`.jsonl`) for compatibility with training pipelines.

### 3. Train/Test Dataset Split with Image Copying (`train_test_split.py`)
- Splits dataset metadata into stratified training and testing subsets.
- Copies corresponding images into separate folders (`train/` and `test/`).
- Saves updated metadata files compatible with downstream training scripts.

### 4. Model Fine-Tuning Script on Google Colab (`gemma_finetuning.py`)
- Loads and prepares training data from Google Drive.
- Formats image-text paired data for vision-language model fine-tuning.
- Configures and runs PEFT LoRA training with the `google/gemma-3-4b-pt` model.
- Supports saving and pushing fine-tuned models to Hugging Face Hub.

### 5. Model Inference Script (`gemma_inference.py`)
- Loads the fine-tuned model and processor from Hugging Face Hub.
- Processes test images for skin lesion classification.
- Generates predictions and saves results to CSV.
- Designed to run on Google Colab with Google Drive integration.

### 6. Performance Evaluation Script (`evaluation.py`)
- Loads ground truth labels and prediction results.
- Calculates accuracy and prints a detailed classification report.
- Generates and displays a confusion matrix heatmap for visual performance analysis.

---

## Requirements

- Python 3.7+
- Key Python packages:
  - `opencv-python`
  - `numpy`
  - `pandas`
  - `scikit-image`
  - `scikit-learn`
  - `transformers`
  - `datasets`
  - `peft`
  - `trl`
  - `torch`
  - `matplotlib`
  - `seaborn`
  - `Pillow`
  - `huggingface_hub`

Install via pip:
```bash
pip install package_names
