# Semantic Segmentation & Domain Shift Analysis (Project B2)

**Course:** Visione Computerizzata e Sistemi Cognitivi  
**Student:** [Ildo Tiberio]  
**Matricola:** [0322500009]

## 📋 Project Overview
This project implements a **U-Net architecture from scratch** to perform binary semantic segmentation on the **Oxford-IIIT Pet Dataset**. 

Beyond standard segmentation, the project focuses on **Domain Shift Analysis**:
1.  **Baseline Training:** Training a model on clean data.
2.  **Robustness Testing:** Evaluating the model on corrupted data (Gaussian noise, blur, color jitter) to demonstrate performance degradation.
3.  **Data Augmentation:** Retraining the model with a heavy augmentation pipeline to recover performance on the corrupted domain.

## 📂 Project Structure

```text
.
├── data/                  # Dataset folder (ignored by git)
├── models/                # Saved model checkpoints (.pth) (also ignored by git)
├── outputs/               # Generated inference images and plots
├── src/
│   ├── unet.py            # Custom U-Net architecture implementation
│   ├── dataset.py         # Oxford-IIIT Pet Dataset wrapper
│   ├── utils.py           # Loss functions (Dice + BCE)
│   ├── train.py           # Baseline training script
│   ├── train_robust.py    # Robust training script (Heavy Augmentation)
│   ├── test_robustness.py # Script to evaluate Domain Shift (Clean vs Corrupted IoU)
│   ├── inference.py       # Generates visual examples for the report
│   └── download_data.py   # Utility to download the dataset
└── README.md
