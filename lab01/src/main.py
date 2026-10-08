import os
import time
import random
import psutil
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_peak_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def measure_training_logreg(X_train, y_train, runs=5):
    times, peak_memories = [], []
    for _ in range(runs):
        set_seeds(42)
        mem_before = get_peak_memory_mb()
        start = time.perf_counter()

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_train)
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_tr_scaled, y_train)

        end = time.perf_counter()
        mem_after = get_peak_memory_mb()

        times.append(end - start)
        peak_memories.append(max(mem_before, mem_after))

    return model, scaler, float(np.median(times)), float(np.median(peak_memories))


def measure_training_rf(X_train, y_train, runs=5):
    times, peak_memories = [], []
    for _ in range(runs):
        set_seeds(42)
        mem_before = get_peak_memory_mb()
        start = time.perf_counter()

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        end = time.perf_counter()
        mem_after = get_peak_memory_mb()

        times.append(end - start)
        peak_memories.append(max(mem_before, mem_after))

    return model, float(np.median(times)), float(np.median(peak_memories))


def measure_inference_single(predict_fn, sample, runs=100):
    times = []
    _ = predict_fn(sample)  # Warm-up
    for _ in range(runs):
        start = time.perf_counter()
        _ = predict_fn(sample)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    return float(np.median(times))


def main():
    set_seeds(42)
    os.makedirs("lab01/results", exist_ok=True)

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    single_sample_raw = X_test[0:1]

    #1. Logistic Regression
    logreg_model, scaler, lr_time, lr_mem = measure_training_logreg(X_train, y_train)
    X_test_scaled = scaler.transform(X_test)
    lr_pred = logreg_model.predict(X_test_scaled)
    lr_acc = round(float(accuracy_score(y_test, lr_pred)), 4)

    lr_path = "lab01/results/model_logisticregression.joblib"
    joblib.dump({"model": logreg_model, "scaler": scaler}, lr_path)
    lr_bytes = os.path.getsize(lr_path)

    lr_latency = measure_inference_single(
        lambda s: logreg_model.predict(scaler.transform(s)), single_sample_raw
    )

    #2. Random Forest
    rf_model, rf_time, rf_mem = measure_training_rf(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = round(float(accuracy_score(y_test, rf_pred)), 4)

    rf_path = "lab01/results/model_randomforest.joblib"
    joblib.dump(rf_model, rf_path)
    rf_bytes = os.path.getsize(rf_path)

    rf_latency = measure_inference_single(rf_model.predict, single_sample_raw)

  
    df_acc = pd.DataFrame([
        {"model": "LogisticRegression", "accuracy": lr_acc},
        {"model": "RandomForest", "accuracy": rf_acc}
    ])
    df_acc.to_csv("lab01/results/baseline_accuracy.csv", index=False)

    df_sys = pd.DataFrame([
        {
            "model": "LogisticRegression",
            "model_size_bytes": lr_bytes,
            "model_size_kb": round(lr_bytes / 1024, 2),
            "train_time_sec": round(lr_time, 4),
            "single_sample_latency_ms": round(lr_latency, 4),
            "peak_memory_mb": round(lr_mem, 2)
        },
        {
            "model": "RandomForest",
            "model_size_bytes": rf_bytes,
            "model_size_kb": round(rf_bytes / 1024, 2),
            "train_time_sec": round(rf_time, 4),
            "single_sample_latency_ms": round(rf_latency, 4),
            "peak_memory_mb": round(rf_mem, 2)
        }
    ])
    df_sys.to_csv("lab01/results/system_metrics.csv", index=False)

    print("--- Baseline Accuracy ---")
    print(df_acc.to_string(index=False))
    print("\n--- System Metrics ---")
    print(df_sys.to_string(index=False))


if __name__ == "__main__":
    main()