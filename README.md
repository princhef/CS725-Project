# Label Smoothing vs No Smoothing in Classification

**CS725: Foundations of Machine Learning — IIT Bombay**

This project presents a systematic experimental study of **label smoothing** as a regularization technique for image classification. We compare standard hard-label training with different label-smoothing factors across multiple CNN architectures and datasets.

## Project Objective

The objective is to study how label smoothing affects:

* Classification accuracy and F1-score
* Model convergence and training dynamics
* Overfitting and generalization
* Prediction confidence
* Model calibration using Expected Calibration Error (ECE)
* The effect of model depth on these trends

## Datasets

The experiments are performed on two standard image-classification benchmarks:

* **MNIST** — 70,000 grayscale handwritten-digit images (28×28), 10 classes
* **CIFAR-10** — 60,000 RGB images (32×32), 10 classes

Standard train/test splits are used for both datasets.

## Experimental Setup

All models are implemented using **PyTorch** and trained using the Adam optimizer with a learning rate of `0.001`.

We compare:

* **No Smoothing:** α = 0
* **Label Smoothing:** α ∈ {0.05, 0.10, 0.15, 0.20, 0.25}

The model architecture is kept fixed while varying only the smoothing factor to ensure a controlled comparison.

### Architectures

| Dataset  | Architectures                         |
| -------- | ------------------------------------- |
| MNIST    | 2-layer CNN, 4-layer CNN              |
| CIFAR-10 | 2-layer CNN, 4-layer CNN, 8-layer CNN |

## Evaluation Metrics

The models are evaluated using:

* **Test Accuracy**
* **F1 Score**
* **Cross-Entropy Loss**
* **Expected Calibration Error (ECE)**
* Training loss and error
* Convergence behaviour
* Overfitting/generalization trends

## Key Findings

The experiments show that label smoothing generally improves model calibration and reduces overconfident predictions.

### CIFAR-10

* Moderate smoothing (**α = 0.05–0.15**) generally provides the best generalization performance.
* **α = 0.10–0.15** typically gives the best calibration.
* Excessive smoothing (**α ≥ 0.20**) can slightly reduce classification performance.
* Label smoothing consistently lowers ECE compared with the hard-label baseline.

### MNIST

* The effect of label smoothing is smaller because MNIST is a comparatively simpler classification task.
* Small smoothing factors (**α = 0.05–0.10**) provide marginal improvements in accuracy stability.
* Calibration still improves consistently with label smoothing.

### Overall Observation

Label smoothing provides a useful trade-off between classification performance and prediction confidence. Moderate smoothing improves generalization while substantially reducing calibration error across different CNN depths.

## Repository Structure

```text
CS725-Project/
│
├── MNIST_2_CNN.py
├── MNIST_4_CNN.py
├── Cifar10_2_CNN.py
├── cifar10_4_CNN.py
├── cifar10_8_CNN.py
└── README.md
```

## Running the Experiments

### 1. Clone the repository

```bash
git clone https://github.com/princhef/CS725-Project.git
cd CS725-Project
```

### 2. Install dependencies

```bash
pip install torch torchvision numpy matplotlib
```

### 3. Run an experiment

For example:

```bash
python MNIST_2_CNN.py
```

or

```bash
python cifar10_8_CNN.py
```

The corresponding scripts train the CNN and evaluate the selected label-smoothing configurations.

## Project Contributors

* **Rudreswar Pal**
* **Pronajit Dey**
* **Priyangshu Dey**
* **Arnob Deb**

## Course

**CS725 — Foundations of Machine Learning**
**Department of Computer Science and Engineering**
**Indian Institute of Technology Bombay**

## Reference

Project report: **Comparison of Label Smoothing vs No Smoothing in Classification**
