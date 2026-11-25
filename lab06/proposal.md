# Individual Project Proposal: Telemetry Fault & Anomaly Classifier

## Student Information
- **Name:** Shcherbakov Oleh
- **Student ID:** 71663
- **Course:** Machine Learning Systems Engineering

## 1. Problem Statement
Build a lightweight neural network classifier that detects operating faults and anomalies from sensor telemetry streams (such as rotational speed, torque, temperature, and wear metrics). The goal is to achieve reliable fault detection with low CPU inference overhead so the model can run in resource-constrained edge or monitoring environments.

## 2. Dataset
- **Source:** Tabular sensor telemetry dataset (such as the AI4I 2020 Predictive Maintenance dataset) containing numerical operating measurements and fault/anomaly labels.
- **Preprocessing:** Feature scaling (`StandardScaler` / normalization) fitted strictly on the training split, train/validation/test split, and class-imbalance handling during training and evaluation.

## 3. Success Metrics
- **Primary Metrics:** **Macro F1-score** and **ROC-AUC** (chosen over raw accuracy alone because fault and anomaly classes are imbalanced relative to normal operation).
- **Secondary Metrics:** Baseline vs. optimized validation F1-score, CPU latency, and serialized model size.

## 4. Deployment Constraints
- **Inference Latency:** $\le 50\text{ ms}$ per sample/batch on CPU.
- **Model Size:** $\le 5\text{ MB}$ for the saved model artifact.

## 5. Planned Optimization Techniques
- **Pruning:** Magnitude-based pruning (`L1` unstructured / structured layer pruning) on dense layers to remove redundant parameters and evaluate accuracy vs. sparsity trade-offs.
- **Quantization:** Post-training quantization (`INT8` dynamic/static quantization in PyTorch or exported runtime) to shrink model size and reduce CPU compute time while preserving fault detection quality.