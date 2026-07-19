# Diabetic Retinopathy Detection

This repository contains code for training, evaluating, and interpreting deep learning models for diabetic retinopathy detection and classification. The project supports a custom lightweight CNN as well as several pretrained architectures.

---

## Supported Models

The following models are available:

- `own_cnn` — Proposed custom CNN model  
- `resnet50`  
- `densenet121`  
- `efficientnet_b0`

---

## Supported Datasets

The following datasets are supported:

- `idrid`
- `eyepacs`

---

## Project Structure
├── config/                  # Experiment and model configuration files (YAML)
│   ├── dataset/             # Dataset-specific configurations
│   ├── models/              # Model-specific configurations
│   └── default.yaml         # Global default configuration
│
├── data/                    # Raw datasets (excluded from version control)
│   ├── EyePACS/
│   └── IDRID/
│       ├── images/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── datasets/                # Dataset loader implementations
│   ├── eyepacs.py
│   └── idrid.py
│
├── preprocessing/           # Image preprocessing pipelines
│   └── ben_graham.py        # Ben Graham retinal preprocessing
│
├── losses/                  # Custom loss functions
│   └── focal_loss.py
│
├── models/                  # Model architectures
│   ├── custom_cnn.py
│   ├── resnet50.py
│   ├── densenet121.py
│   └── efficientnetb0.py
│
├── visualization/           # Model explainability & interpretability tools
│   ├── cam.py               # Class Activation Mapping (CAM)
│   ├── gradcam.py           # Grad-CAM
│   ├── guided_backprop.py   # Guided Backpropagation
│   └── guided_gradcam.py    # Guided Grad-CAM
│
├── scripts/                 # Batch scripts for running experiments (excluded from version control)
│   ├── eyepacs_own_cnn.sh
│   ├── eyepacs_pretrained_basic_cnn.sh
│   ├── finetuning_basic_cnn_on_IDRID.sh
│   ├── gradcam_basic_cnn.sh
│   ├── guided_backprop_basic_cnn.sh
│   └── guided_gradcam_basic_cnn.sh
│
├── main.py                  # Training entry point
├── eval.py                  # Evaluation script
├── evaluator.py             # Evaluation utilities and metrics
├── README.md
└── requirements.txt

## Training

To train a model, both the model architecture and dataset must be specified. Training behavior can be customized using configuration keys.

### Required Options
- `model` — Model architecture to use  
- `dataset` — Dataset to train on  

### Optional Overrides
- `dataset.path` — Path to the dataset directory 
- `checkpoint.path` — Path to the checkpoint for evaluation  
- `checkpoint.paths` — Paths to the checkpoints for ensemble mode evaluation 
- `checkpoint_dir` — Directory where model checkpoints will be saved  
- `dataset.batch_size` — Training batch size  
- `epochs` — Number of training epochs  
- `lr` — Learning rate  
- `weight_decay` — Weight decay coefficient  
- `model.dropout_rate` — Dropout rate for the model  

### Example (Training)
python train.py \
  model=own_cnn \
  dataset=idrid \
  dataset.path=/path/to/idrid \
  dataset.batch_size=16 \
  epochs=50 \
  lr=1e-3 \
  weight_decay=1e-4 \
  model.dropout_rate=0.5 \
  checkpoint_dir=./checkpoints


### Example (Ensemble Evaluation)
  python eval.py \
  ensemble_mode=True \
  dataset=idrid \
  ensemble_mode=True\
  checkpoint_paths="[ckpt1.ckpt, ckpt2.ckpt, ckpt3.ckpt]" \
  +model.num_base_filters=32

### Example (Evaluation)
  python eval.py \
  ensemble_mode=True \
  dataset=idrid \
  ensemble_mode=True\
  model=own_cnn\
  checkpoint_paths="ckpt" \
  model.num_base_filters=32

### Example (Visualization)
  python visualize.py \
  model=own_cnn \
  dataset=idrid \
  dataset.path=/path/to/idrid \
  checkpoint_path=./checkpoints/best_model.ckpt

### Example (Model Summary)
  python model_summary.py \
  model=own_cnn


