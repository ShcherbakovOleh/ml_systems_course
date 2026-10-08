\# Laboratory Work #1 Report



\## 1. Environment \& Setup

\- \*\*Python Version:\*\* 3.11.9

\- \*\*OS:\*\* Windows

\- \*\*Dependencies:\*\* Installed according to `requirements.txt` (including `scikit-learn`, `torch`, `mlflow`, `pyarrow==15.0.2`, etc.)



\## 2. Model Metrics



\### Baseline Accuracy

| Model | Accuracy |

| :--- | :--- |

| LogisticRegression | 0.9708 |

| RandomForest | 0.9357 |



\### System Metrics

| Model | Model Size (KB) | Train Time (s) | Single Latency (ms) | Peak Memory (MB) |

| :--- | :--- | :--- | :--- | :--- |

| LogisticRegression | \~1.85 | \~0.0250 | \~0.1100 | \~265.00 |

| RandomForest | \~284.07 | \~0.2700 | \~2.7000 | \~265.60 |



\## 3. Analysis \& Trade-offs

1\. \*\*Accuracy vs. Speed:\*\* Logistic Regression with feature scaling achieves higher accuracy on this dataset (0.9708 vs. 0.9357) while executing significantly faster during both training and single-sample inference.

2\. \*\*Model Footprint:\*\* The serialized `LogisticRegression` model takes only \~1.85 KB of disk space, whereas the `RandomForest` ensemble requires around 284 KB (over 150x larger).

3\. \*\*Inference Latency:\*\* For single-sample predictions, Logistic Regression responds in \~0.11 ms compared to \~2.70 ms for Random Forest. This makes Logistic Regression far more suitable for high-throughput, low-latency production environments with strict SLA requirements.

