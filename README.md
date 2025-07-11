<div align="center"><h1>Handwriting ML – Symbol Recognition Using CNN</h1></div>

This project implements a deep learning pipeline to detect and classify handwritten mathematical symbols. Built with PyTorch and executed in Google Colab, it was developed as part of a campus workshop and hackathon. The solution secured the **runner-up position** in the [Torch It Up](https://www.kaggle.com/competitions/torch-it-up) Kaggle competition.

---

## Project Overview

- **Objective**: Classify grayscale images of handwritten mathematical symbols (369 classes).
- **Framework**: PyTorch
- **Execution Environment**: Google Colab
- **Dataset**: Provided by the Torch It Up Kaggle competition
- **Deliverables**: Trained model weights (`symbol_classifier.pth`) and prediction file (`submission.csv`)

---

## Repository Structure

```

.
├── main.ipynb                  # Full training and inference pipeline
├── data/
│   ├── train/                  # Training images
│   ├── test/                   # Test images
│   ├── train.csv               # Image IDs with labels
│   └── sample_submission.csv   # Submission format
├── torch-it-up.zip             # Original compressed dataset
├── symbol_classifier.pth       # Trained model weights
└── submission.csv              # Output predictions (generated)

````

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/s4nj1th/handwriting-ml.git
cd handwriting-ml
````

### 2. Prepare the Dataset

You do **not** need to manually extract the dataset. If `torch-it-up.zip` is placed in the root directory, it will be automatically extracted by a cell in `main.ipynb`.

Expected structure after extraction:

```
data/
├── train/
├── test/
├── train.csv
└── sample_submission.csv
```

---

## Model Workflow

The entire pipeline is implemented in [`main.ipynb`](./main.ipynb). Open it in **Google Colab** or a local Jupyter environment to reproduce training and inference.

### Steps Overview:

1. **Data Loading & Visualization**

   * Loads images and labels using `pandas` and `os`
   * Displays class distributions and sample images

2. **Custom Dataset Class**

   * `SymbolDataset` handles image preprocessing:

     * Resize to `32x32`
     * Convert to grayscale
     * Normalize (`mean=0.5`, `std=0.5`)

3. **Model Architecture**

   * CNN with:

     * Two convolutional blocks (Conv → ReLU → MaxPool)
     * Two fully connected layers with dropout
     * Final output layer for 369 classes

4. **Training**

   * Optimizer: Adam (`lr=0.001`)
   * Loss: CrossEntropyLoss
   * Batch size: 64
   * Epochs: 10
   * Training/validation accuracy and loss logged per epoch

5. **Inference**

   * Loads test images and runs predictions
   * Outputs predictions to `submission.csv`
   * Saves model weights to `symbol_classifier.pth`

---

## Methodology

### Data Pipeline

* CSV label files mapped to image directories
* PyTorch `Dataset` class used for batch preprocessing
* `DataLoader` used for training and testing splits

### Preprocessing

* Images resized to 32×32 pixels
* Converted to single-channel (grayscale)
* Normalized to zero mean and unit variance

### CNN Design

* **Conv Layers**:

  * `Conv2D (1 → 32)` → ReLU → MaxPool
  * `Conv2D (32 → 64)` → ReLU → MaxPool
* **Fully Connected Layers**:

  * Flatten → Linear(64×8×8 → 512) → Dropout
  * Linear(512 → 369)

### Training Loop

* Loss: CrossEntropy
* Optimizer: Adam
* Epochs: 10
* Outputs training and validation metrics

### Inference & Export

* Generates prediction probabilities
* Converts outputs to class labels
* Exports results to `submission.csv`

---

## Results and Future Work

### Evaluation

* Accuracy used for model evaluation during validation

### Areas for Improvement

* Data Augmentation: Rotation, flips, brightness/contrast shifts
* Model Architecture: Transfer learning using deeper CNNs (e.g., ResNet)
* Hyperparameter Tuning: Learning rate, regularization, batch size
* Larger Dataset: Better generalization and reduced overfitting

---

## Team

### Sanjith

* [GitHub](https://github.com/s4nj1th)
* [LinkedIn](https://www.linkedin.com/in/s4nj1th)
* [Twitter (X)](https://x.com/s4nj1th)
* [Email](mailto:sanjith.develops@gmail.com)

### Sanjai P G

* [GitHub](https://github.com/SanjaiPG)
* [LinkedIn](https://www.linkedin.com/in/sanjai-p-g-a079632a6)
* [Instagram](https://www.instagram.com/_sanjai_pgs)
* [Email](mailto:sanjaigukesan8686@gmail.com)

