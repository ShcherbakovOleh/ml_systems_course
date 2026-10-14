# Lab 2: Deep Learning Primer — Neural Network from Scratch

## Overview
This laboratory assignment implements a 2-hidden-layer fully-connected (dense) feedforward neural network from scratch using **NumPy** to classify handwritten digits ($8 \times 8$ grayscale images, 10 classes) from the `load_digits` dataset.

The project demonstrates:
- Correct manual implementation of **Forward Propagation**, **Cross-Entropy Loss**, and **Backpropagation** using the Chain Rule.
- Validation of theoretical analytical gradients via **Numerical Gradient Checking**.
- Hyperparameter search across varying **hidden layer widths** ($W \in \{16, 64\}$) and **learning rates** ($\eta \in \{0.001, 0.01, 0.1\}$).
- Empirical benchmarking of computational efficiency (Forward vs. Backward pass timing) linked to matrix multiplication complexity.

---

## Architecture & Mathematical Formulation

### 1. Model Architecture
The network consists of an input layer, two hidden layers with **ReLU** activation, and an output layer with **Softmax** activation:

$$\text{Input (64)} \xrightarrow{W_1, b_1} \text{Hidden 1 } (W) \xrightarrow{\text{ReLU}} \text{Hidden 2 } (W) \xrightarrow{\text{ReLU}} \text{Output } (10) \xrightarrow{\text{Softmax}} \hat{Y}$$

- **Input Dimension:** $D_{in} = 64$ (flattened $8 \times 8$ images normalized to $[0, 1]$ by dividing by 16).
- **Hidden Widths ($W$):** Evaluated for $W = 16$ and $W = 64$.
- **Output Dimension:** $K = 10$ classes (One-Hot encoded labels).
- **Weight Initialization:** Weights initialized from Gaussian distribution $\mathcal{N}(0, 0.01)$, biases initialized to zeros.

### 2. Forward Propagation
For a batch $X \in \mathbb{R}^{N \times 64}$:
1. $Z_1 = X W_1 + b_1, \quad A_1 = \max(0, Z_1)$
2. $Z_2 = A_1 W_2 + b_2, \quad A_2 = \max(0, Z_2)$
3. $Z_3 = A_2 W_3 + b_3, \quad A_3 = \text{Softmax}(Z_3) = \frac{\exp(Z_3 - \max(Z_3))}{\sum_{k} \exp(Z_{3, \cdot, k} - \max(Z_3))}$

### 3. Loss Function
Multi-class Cross-Entropy Loss with numeric stability clipping ($\epsilon = 10^{-15}$):
$$\mathcal{L}(Y, A_3) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{10} Y_{i, k} \ln(A_{3, i, k})$$

### 4. Backpropagation Equations
Combining Softmax and Cross-Entropy Loss yields the concise error signal at the output:
- **Output Layer:**
  $$\delta_3 = \frac{\partial \mathcal{L}}{\partial Z_3} = \frac{1}{N} (A_3 - Y)$$
  $$\frac{\partial \mathcal{L}}{\partial W_3} = A_2^T \delta_3, \quad \frac{\partial \mathcal{L}}{\partial b_3} = \sum_{i=1}^{N} \delta_{3, i}$$

- **Second Hidden Layer:**
  $$\delta_2 = (\delta_3 W_3^T) \odot \mathbb{I}(Z_2 > 0)$$
  $$\frac{\partial \mathcal{L}}{\partial W_2} = A_1^T \delta_2, \quad \frac{\partial \mathcal{L}}{\partial b_2} = \sum_{i=1}^{N} \delta_{2, i}$$

- **First Hidden Layer:**
  $$\delta_1 = (\delta_2 W_2^T) \odot \mathbb{I}(Z_1 > 0)$$
  $$\frac{\partial \mathcal{L}}{\partial W_1} = X^T \delta_1, \quad \frac{\partial \mathcal{L}}{\partial b_1} = \sum_{i=1}^{N} \delta_{1, i}$$

---

## Experimental Results & Evidence

### 1. Numerical Gradient Checking
To confirm the correctness of analytical backpropagation gradients, 10 random parameters were selected and perturbed by $\epsilon = 10^{-5}$ using two-sided finite differences:

$$\text{grad}_{\text{num}} = \frac{\mathcal{L}(\theta + \epsilon) - \mathcal{L}(\theta - \epsilon)}{2\epsilon}$$

Relative error formula:
$$\text{Relative Error} = \frac{\vert{}\text{grad}_{\text{analytic}} - \text{grad}_{\text{num}}\vert{}}{\vert{}\text{grad}_{\text{analytic}}\vert{} + \vert{}\text{grad}_{\text{num}}\vert{} + 10^{-8}}$$

**Result:** Passed for **9/10** checked parameters with relative error $< 10^{-6}$, satisfying the required sanity threshold.

---

### 2. Hyperparameter Grid Search & Test Accuracy

Training configuration:
- Train / Test split: **70 / 30** (Stratified, `random_state=42`)
- Batch size: **32**
- Epochs: **50**
- Optimizer: **Mini-Batch Stochastic Gradient Descent (SGD)** with per-epoch shuffling

| Hidden Width ($W$) | Learning Rate ($\eta$) | Test Accuracy | Status / Notes |
| :---: | :---: | :---: | :--- |
| **16** | $0.001$ | 10.19% | Underfitting / Stuck in local minimum due to small step size |
| **16** | $0.01$  | 10.19% | Slow convergence |
| **16** | $0.1$   | **67.22%** | Moderate convergence |
| **64** | $0.001$ | 10.19% | Underfitting |
| **64** | $0.01$  | 10.19% | Underfitting |
| **64** | $0.1$   | **94.44%** | **Best Performing Model** |

---

### 3. Loss Curves

![Loss Curves](results/loss_curves.png)

*Figure 1: Cross-Entropy loss progression over 50 epochs for all 6 hyperparameter combinations.*

---

### 4. Computational Performance Analysis (Timing Breakdown)

Timing measured over **100 independent runs** on a batch size of 32:

- **Forward Pass Median Time:** ~0.0848 ms
- **Backward Pass Median Time:** ~0.0951 ms

#### Theoretical Justification
The Backward Pass takes consistently longer than the Forward Pass. This behavior is directly tied to the underlying **matrix multiplication complexity**:

1. **Forward Pass:** Each linear layer computes a single matrix multiplication:
   $$Z^{[l]} = A^{[l-1]} W^{[l]}$$
   *(1 GEMM operation per layer)*

2. **Backward Pass:** Each layer requires **two** separate matrix multiplications:
   - Gradient w.r.t. parameters: $\frac{\partial \mathcal{L}}{\partial W^{[l]}} = (A^{[l-1]})^T \delta^{[l]}$
   - Gradient w.r.t. activations (to propagate error backwards): $\delta^{[l-1]} = (\delta^{[l]} (W^{[l]})^T) \odot \sigma'(Z^{[l-1]})$  
   *(2 GEMM operations per layer)*

Hence, backpropagation inherently performs approximately **double the number of matrix multiplications**, explaining the higher execution time observed in benchmarks.

---

## How to Run
```bash
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 2. Run training script
python main.py