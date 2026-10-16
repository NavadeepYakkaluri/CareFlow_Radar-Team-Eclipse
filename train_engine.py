#!/usr/bin/env python3
"""
CAREFLOW RADAR: Machine Learning Pipeline for Hospital Patient-Flow
Track: Health - "The Forgotten Patient" (Datathon 2026)

Trained strictly from scratch on provided datasets:
- patients-2.csv (2,000 patients)
- encounters.csv (8,000 visits)
- departments.csv (12 departments)
- diagnostic_tests.csv (5,000 tests)

Evaluated with strict Time-Aware Train/Test Split (no future leakage).
"""

import os
import json
import math
import warnings
from datetime import datetime
import numpy as np
import pandas as pd

# Suppress Apple Accelerate BLAS benign matmul floating-point warning on macOS arm64
warnings.filterwarnings('ignore', category=RuntimeWarning)

def load_and_preprocess_data():
    print("[1/5] Loading datasets...")
    # Load raw CSVs
    departments = pd.read_csv("departments.csv")
    encounters = pd.read_csv("encounters.csv")
    tests = pd.read_csv("diagnostic_tests.csv")
    
    # Check for patient filename
    patient_file = "patients-2.csv" if os.path.exists("patients-2.csv") else "patients.csv"
    patients = pd.read_csv(patient_file)
    
    print(f"  Loaded: {len(departments)} departments, {len(patients)} patients, "
          f"{len(encounters)} encounters, {len(tests)} diagnostic tests.")

    # Parse timestamps
    print("[2/5] Cleaning and feature engineering across tables...")
    encounters['arrival_time'] = pd.to_datetime(encounters['arrival_time'])
    encounters['service_start_time'] = pd.to_datetime(encounters['service_start_time'])
    encounters['service_end_time'] = pd.to_datetime(encounters['service_end_time'])
    encounters['discharge_time'] = pd.to_datetime(encounters['discharge_time'])

    tests['requested_time'] = pd.to_datetime(tests['requested_time'])
    tests['completion_time'] = pd.to_datetime(tests['completion_time'])
    tests['tat_minutes'] = (tests['completion_time'] - tests['requested_time']).dt.total_seconds() / 60.0

    # Diagnostic test aggregation per encounter
    test_agg = tests.groupby('encounter_id').agg(
        total_tests=('test_id', 'count'),
        urgent_tests=('priority', lambda s: (s == 'Urgent').sum()),
        avg_test_tat=('tat_minutes', 'mean'),
        max_test_tat=('tat_minutes', 'max'),
        has_imaging=('test_type', lambda s: s.isin(['CT scan', 'MRI', 'X-ray']).any().astype(int)),
        has_lab=('test_type', lambda s: s.isin(['Blood test', 'Urine test']).any().astype(int))
    ).reset_index()

    # Join encounters with tests
    df = pd.merge(encounters, test_agg, on='encounter_id', how='left')
    df['total_tests'] = df['total_tests'].fillna(0).astype(int)
    df['urgent_tests'] = df['urgent_tests'].fillna(0).astype(int)
    df['avg_test_tat'] = df['avg_test_tat'].fillna(0.0)
    df['max_test_tat'] = df['max_test_tat'].fillna(0.0)
    df['has_imaging'] = df['has_imaging'].fillna(0).astype(int)
    df['has_lab'] = df['has_lab'].fillna(0).astype(int)

    # Join with departments
    df = pd.merge(df, departments, on='department_id', how='left')

    # Join with patients
    df = pd.merge(df, patients, on='patient_id', how='left')

    # Chronological sort for strict time-aware operations
    df = df.sort_values('arrival_time').reset_index(drop=True)

    # Journey metrics
    df['post_service_lag'] = (df['discharge_time'] - df['service_end_time']).dt.total_seconds() / 60.0
    df['total_journey_minutes'] = (df['discharge_time'] - df['arrival_time']).dt.total_seconds() / 60.0

    # Time features
    df['arrival_hour'] = df['arrival_time'].dt.hour
    df['arrival_dayofweek'] = df['arrival_time'].dt.dayofweek
    df['is_weekend'] = (df['arrival_dayofweek'] >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['arrival_hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['arrival_hour'] / 24.0)

    # Dynamic Queue Depth at Arrival (Simulating real-time queue load)
    # Active patient in department d at time t: arrived <= t and discharge > t
    print("  Computing dynamic queue depth at arrival for each department...")
    arrival_arr = df['arrival_time'].values
    discharge_arr = df['discharge_time'].values
    dept_arr = df['department_id'].values

    n = len(df)
    active_load = np.zeros(n, dtype=int)
    
    # Efficient windowed check: since data is sorted by arrival_time,
    # look back at recent visits within 24 hours
    for i in range(n):
        curr_arr = arrival_arr[i]
        curr_dept = dept_arr[i]
        # Count preceding patients in same dept whose discharge is after curr_arr
        lookback = max(0, i - 150)
        c = 0
        for j in range(lookback, i):
            if dept_arr[j] == curr_dept and discharge_arr[j] > curr_arr:
                c += 1
        active_load[i] = c

    df['queue_depth_at_arrival'] = active_load
    df['staff_load_ratio'] = df['queue_depth_at_arrival'] / np.maximum(df['staff_count'], 1)
    df['bed_load_ratio'] = df['queue_depth_at_arrival'] / np.maximum(df['bed_count'], 1)

    # Severe Delay Indicator (>60 min wait time or >180 min total journey)
    df['is_severe_delay'] = ((df['wait_minutes'] > 60) | (df['total_journey_minutes'] > 180)).astype(int)

    return df, departments, patients, encounters, tests


class FromScratchRidgeRegressor:
    """
    Closed-Form Regularized Ridge Regression trained strictly from scratch with NumPy.
    w = (X^T X + alpha * I)^(-1) X^T y
    Includes feature standardization and exact coefficient interpretability.
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None
        self.bias = None
        self.mean = None
        self.std = None
        self.feature_names = []

    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names if feature_names else [f"f_{i}" for i in range(X.shape[1])]
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std < 1e-5] = 1.0  # Prevent division by zero
        X_norm = np.clip((X - self.mean) / self.std, -10.0, 10.0)
        N, D = X_norm.shape

        # Add bias column
        X_design = np.hstack([np.ones((N, 1)), X_norm])
        I = np.eye(D + 1)
        I[0, 0] = 0.0  # Do not regularize bias

        # Solve (X^T X + alpha * I) w = X^T y
        A = X_design.T @ X_design + self.alpha * I
        b = X_design.T @ y
        w_full = np.linalg.solve(A, b)

        self.bias = w_full[0]
        self.weights = w_full[1:]
        return self

    def predict(self, X):
        X_norm = np.clip((X - self.mean) / self.std, -10.0, 10.0)
        return X_norm @ self.weights + self.bias

    def get_feature_importances(self):
        total = np.sum(np.abs(self.weights))
        if total == 0:
            total = 1.0
        normalized_importance = np.abs(self.weights) / total
        return sorted(zip(self.feature_names, self.weights, normalized_importance),
                      key=lambda x: abs(x[1]), reverse=True)


class FromScratchLogisticClassifier:
    """
    Logistic Regression with L2 Regularization trained with Gradient Descent from scratch.
    p = 1 / (1 + exp(-z))
    """
    def __init__(self, lr=0.05, alpha=0.1, n_iter=300):
        self.lr = lr
        self.alpha = alpha
        self.n_iter = n_iter
        self.weights = None
        self.bias = 0.0
        self.mean = None
        self.std = None

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))

    def fit(self, X, y):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std < 1e-5] = 1.0

        X_norm = np.clip((X - self.mean) / self.std, -10.0, 10.0)
        N, D = X_norm.shape

        self.weights = np.zeros(D)
        self.bias = 0.0

        for _ in range(self.n_iter):
            z = X_norm @ self.weights + self.bias
            p = self._sigmoid(z)
            error = p - y

            # Gradients with L2 regularization
            grad_w = (X_norm.T @ error) / N + (self.alpha / N) * self.weights
            grad_b = np.mean(error)

            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b
        return self

    def predict_proba(self, X):
        X_norm = np.clip((X - self.mean) / self.std, -10.0, 10.0)
        z = X_norm @ self.weights + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


class FromScratchDecisionStumpEnsemble:
    """
    Gradient Boosted Stumps Regressor built from mathematical first principles in NumPy.
    Iteratively fits shallow decision boundaries on residuals to capture non-linear interactions.
    """
    def __init__(self, n_estimators=30, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.base_pred = 0.0
        self.stumps = []

    def fit(self, X, y):
        self.base_pred = np.mean(y)
        residuals = y - self.base_pred

        N, D = X.shape
        for _ in range(self.n_estimators):
            best_feat = 0
            best_thresh = 0.0
            best_left_val = 0.0
            best_right_val = 0.0
            best_loss = float('inf')

            # Random sample of features to speed up and diversify
            feats = np.random.choice(D, size=min(8, D), replace=False)
            for f_idx in feats:
                col = X[:, f_idx]
                percentiles = np.percentile(col, [20, 40, 60, 80])
                for t in percentiles:
                    mask_left = col <= t
                    mask_right = ~mask_left
                    if not np.any(mask_left) or not np.any(mask_right):
                        continue
                    left_val = np.mean(residuals[mask_left])
                    right_val = np.mean(residuals[mask_right])
                    pred = np.where(mask_left, left_val, right_val)
                    loss = np.sum((residuals - pred) ** 2)
                    if loss < best_loss:
                        best_loss = loss
                        best_feat = f_idx
                        best_thresh = t
                        best_left_val = left_val
                        best_right_val = right_val

            self.stumps.append((best_feat, best_thresh, best_left_val, best_right_val))
            # Update residuals
            col_b = X[:, best_feat]
            update = np.where(col_b <= best_thresh, best_left_val, best_right_val)
            residuals -= self.lr * update
        return self

    def predict(self, X):
        preds = np.full(len(X), self.base_pred)
        for feat, thresh, left_val, right_val in self.stumps:
            col = X[:, feat]
            update = np.where(col <= thresh, left_val, right_val)
            preds += self.lr * update
        return preds


def train_and_evaluate(df):
    print("[3/5] Setting up Time-Aware Split and Encoding Features...")
    
    # Define features
    feature_cols = [
        'staff_count', 'bed_count', 'daily_capacity',
        'age', 'total_tests', 'urgent_tests', 'avg_test_tat',
        'has_imaging', 'has_lab', 'queue_depth_at_arrival',
        'staff_load_ratio', 'bed_load_ratio',
        'hour_sin', 'hour_cos', 'is_weekend'
    ]

    # One-hot encode categorical features cleanly with pandas
    cats = ['triage_level', 'visit_type', 'chronic_condition', 'insurance_type', 'service_type']
    df_encoded = pd.get_dummies(df[feature_cols + cats], drop_first=True, dtype=float)
    final_features = list(df_encoded.columns)

    X = df_encoded.values
    y_reg = df['wait_minutes'].values.astype(float)
    y_clf = df['is_severe_delay'].values.astype(float)
    arrival_times = df['arrival_time'].values

    # STRICT TIME-AWARE SPLIT (80% train, 20% test chronologically)
    split_idx = int(0.80 * len(df))
    split_date = pd.to_datetime(arrival_times[split_idx])
    print(f"  Strict Time-Aware Split Date: {split_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Train Set: {split_idx} visits ({arrival_times[0]} to {split_date})")
    print(f"  Test Set:  {len(df) - split_idx} visits ({split_date} to {arrival_times[-1]})")

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train_reg, y_test_reg = y_reg[:split_idx], y_reg[split_idx:]
    y_train_clf, y_test_clf = y_clf[:split_idx], y_clf[split_idx:]

    # 1. BASELINE EVALUATION
    # Baseline 1: Global Mean Baseline
    mean_val = np.mean(y_train_reg)
    base_preds = np.full_like(y_test_reg, mean_val)
    base_mae = np.mean(np.abs(y_test_reg - base_preds))
    base_rmse = np.sqrt(np.mean((y_test_reg - base_preds) ** 2))
    base_r2 = 0.0

    # Baseline 2: Triage-Only Group Mean
    triage_map = df.iloc[:split_idx].groupby('triage_level')['wait_minutes'].mean().to_dict()
    triage_test_preds = df.iloc[split_idx:]['triage_level'].map(triage_map).fillna(mean_val).values
    triage_mae = np.mean(np.abs(y_test_reg - triage_test_preds))
    triage_rmse = np.sqrt(np.mean((y_test_reg - triage_test_preds) ** 2))
    triage_ss_res = np.sum((y_test_reg - triage_test_preds) ** 2)
    triage_ss_tot = np.sum((y_test_reg - mean_val) ** 2)
    triage_r2 = 1.0 - (triage_ss_res / triage_ss_tot)

    print(f"\n  Baseline 1 (Global Mean):  MAE={base_mae:.2f}m, RMSE={base_rmse:.2f}m, R2={base_r2:.4f}")
    print(f"  Baseline 2 (Triage-Only):  MAE={triage_mae:.2f}m, RMSE={triage_rmse:.2f}m, R2={triage_r2:.4f}")

    # 2. FROM-SCRATCH RIDGE REGRESSION
    print("\n[4/5] Training From-Scratch ML Models...")
    np.random.seed(42)
    ridge_model = FromScratchRidgeRegressor(alpha=10.0)
    ridge_model.fit(X_train, y_train_reg, feature_names=final_features)
    ridge_preds = ridge_model.predict(X_test)

    r_mae = np.mean(np.abs(y_test_reg - ridge_preds))
    r_rmse = np.sqrt(np.mean((y_test_reg - ridge_preds) ** 2))
    r_ss_res = np.sum((y_test_reg - ridge_preds) ** 2)
    r_ss_tot = np.sum((y_test_reg - mean_val) ** 2)
    r_r2 = 1.0 - (r_ss_res / r_ss_tot)

    print(f"  Ridge Regressor (From Scratch): MAE={r_mae:.2f}m, RMSE={r_rmse:.2f}m, R2={r_r2:.4f}")

    # 3. FROM-SCRATCH GRADIENT BOOSTED ENSEMBLE
    gbdt_model = FromScratchDecisionStumpEnsemble(n_estimators=40, learning_rate=0.08)
    gbdt_model.fit(X_train, y_train_reg)
    gbdt_preds = gbdt_model.predict(X_test)

    g_mae = np.mean(np.abs(y_test_reg - gbdt_preds))
    g_rmse = np.sqrt(np.mean((y_test_reg - gbdt_preds) ** 2))
    g_ss_res = np.sum((y_test_reg - gbdt_preds) ** 2)
    g_r2 = 1.0 - (g_ss_res / r_ss_tot)

    print(f"  GBDT Ensemble (From Scratch):   MAE={g_mae:.2f}m, RMSE={g_rmse:.2f}m, R2={g_r2:.4f}")

    # 4. CLASSIFICATION MODEL (Probability of Severe Delay)
    clf_model = FromScratchLogisticClassifier(lr=0.1, alpha=0.5, n_iter=400)
    clf_model.fit(X_train, y_train_clf)
    clf_probs = clf_model.predict_proba(X_test)
    clf_preds = (clf_probs >= 0.45).astype(int)

    # Classification Metrics
    tp = np.sum((clf_preds == 1) & (y_test_clf == 1))
    fp = np.sum((clf_preds == 1) & (y_test_clf == 0))
    fn = np.sum((clf_preds == 0) & (y_test_clf == 1))
    tn = np.sum((clf_preds == 0) & (y_test_clf == 0))

    accuracy = (tp + tn) / len(y_test_clf)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * (precision * recall) / max(precision + recall, 1e-6)

    # Simplified Mann-Whitney ROC-AUC approximation
    pos_probs = clf_probs[y_test_clf == 1]
    neg_probs = clf_probs[y_test_clf == 0]
    # Sample rank comparison for AUC
    n_samples = min(2000, len(pos_probs) * len(neg_probs))
    sample_pos = np.random.choice(pos_probs, size=min(1000, len(pos_probs)))
    sample_neg = np.random.choice(neg_probs, size=min(1000, len(neg_probs)))
    auc_est = np.mean(sample_pos[:, None] > sample_neg[None, :])

    print(f"\n  Severe Delay Classifier:")
    print(f"    Accuracy:  {accuracy*100:.1f}%")
    print(f"    Precision: {precision*100:.1f}%")
    print(f"    Recall:    {recall*100:.1f}%")
    print(f"    F1 Score:  {f1:.3f}")
    print(f"    Est. AUC:  {auc_est:.3f}")

    # Extract Top Root Cause Signals (Feature Attributions)
    top_features = ridge_model.get_feature_importances()[:8]
    print("\n  Top Explainability Signals (Root Cause Drivers):")
    for feat, weight, imp in top_features:
        direction = "INCREASES WAIT" if weight > 0 else "REDUCES WAIT"
        print(f"    - {feat:30s} (Weight: {weight:+.2f}, Impact: {imp*100:.1f}% - {direction})")

    # Generate Operational Artifacts for Prototype
    print("\n[5/5] Generating Prototype JSON Artifacts...")
    
    # 1. Department KPIs
    dept_kpis = []
    for d_id, group in df.groupby('department_id'):
        d_row = group.iloc[0]
        avg_wait = group['wait_minutes'].mean()
        avg_journey = group['total_journey_minutes'].mean()
        severe_rate = group['is_severe_delay'].mean()
        post_lag = group['post_service_lag'].mean()
        dept_kpis.append({
            "department_id": d_id,
            "department_name": d_row['department_name'],
            "service_type": d_row['service_type'],
            "staff_count": int(d_row['staff_count']),
            "bed_count": int(d_row['bed_count']),
            "daily_capacity": int(d_row['daily_capacity']),
            "avg_wait_minutes": round(avg_wait, 1),
            "avg_journey_minutes": round(avg_journey, 1),
            "post_service_lag_avg": round(post_lag, 1),
            "severe_delay_rate": round(severe_rate * 100, 1),
            "total_visits": len(group)
        })

    # Sort by avg_wait descending
    dept_kpis = sorted(dept_kpis, key=lambda x: x['avg_wait_minutes'], reverse=True)

    # 2. Priority Investigation Queue (Answering "What should ops examine first?")
    # Formulate rank = Delay Probability * Queue Strain * Vulnerability
    recent_encounters = df.iloc[-40:].copy()  # Most recent visits in test set
    recent_X = df_encoded.iloc[-40:].values
    recent_pred_waits = ridge_model.predict(recent_X)
    recent_delay_probs = clf_model.predict_proba(recent_X)

    investigation_queue = []
    for idx, (_, row) in enumerate(recent_encounters.iterrows()):
        prob = float(recent_delay_probs[idx])
        pred_wait = float(recent_pred_waits[idx])
        
        # Operational Risk / Impact calculation
        impact_score = round(prob * (row['queue_depth_at_arrival'] + (row['age'] > 65) * 3 + (row['urgent_tests'] > 0) * 4), 1)
        
        # Primary Root Cause Signal
        if row['queue_depth_at_arrival'] >= 5:
            root_cause = f"Department Congestion (Queue depth {row['queue_depth_at_arrival']})"
            rec_action = f"Deploy floating nurse to {row['department_name']}; divert non-emergency walk-ins."
        elif row['total_tests'] >= 2:
            root_cause = f"Diagnostic Lag ({row['total_tests']} tests requested; TAT ~{row['avg_test_tat']:.0f}m)"
            rec_action = f"Expedite lab/imaging batch reading; place on priority alert."
        elif row['triage_level'] == 'Low':
            root_cause = "Low Triage Starvation (Delayed by higher severity cases)"
            rec_action = "Fast-track simple protocol to prevent 70+ min drift."
        else:
            root_cause = "Staffing Capacity Ceiling Reached"
            rec_action = "Coordinate with bed manager for accelerated discharge."

        investigation_queue.append({
            "encounter_id": row['encounter_id'],
            "patient_id": row['patient_id'],
            "department_id": row['department_id'],
            "department_name": row['department_name'],
            "arrival_time": str(row['arrival_time']),
            "triage_level": row['triage_level'],
            "visit_type": row['visit_type'],
            "age": int(row['age']),
            "chronic_condition": str(row['chronic_condition']),
            "predicted_wait_min": round(pred_wait, 1),
            "delay_probability": round(prob, 2),
            "impact_score": impact_score,
            "root_cause": root_cause,
            "recommended_action": rec_action,
            "total_tests": int(row['total_tests'])
        })

    # Sort queue by Impact Score descending
    investigation_queue = sorted(investigation_queue, key=lambda x: x['impact_score'], reverse=True)

    # 3. Hourly Delay Pattern
    hourly_stats = df.groupby('arrival_hour').agg(
        avg_wait=('wait_minutes', 'mean'),
        patient_volume=('encounter_id', 'count'),
        severe_delays=('is_severe_delay', 'sum')
    ).reset_index()

    hourly_patterns = []
    for _, row in hourly_stats.iterrows():
        hourly_patterns.append({
            "hour": int(row['arrival_hour']),
            "avg_wait": round(row['avg_wait'], 1),
            "patient_volume": int(row['patient_volume']),
            "severe_delays": int(row['severe_delays'])
        })

    # 4. Diagnostic Turnaround Monitor
    test_tat_summary = []
    for t_type, t_group in df[df['total_tests'] > 0].groupby('has_imaging'):
        pass

    # Re-read raw tests for summary
    raw_tests = pd.read_csv("diagnostic_tests.csv")
    raw_tests['req'] = pd.to_datetime(raw_tests['requested_time'])
    raw_tests['comp'] = pd.to_datetime(raw_tests['completion_time'])
    raw_tests['tat'] = (raw_tests['comp'] - raw_tests['req']).dt.total_seconds() / 60.0

    tat_summary = []
    for t_type, group in raw_tests.groupby('test_type'):
        urgent_grp = group[group['priority'] == 'Urgent']
        routine_grp = group[group['priority'] == 'Routine']
        tat_summary.append({
            "test_type": t_type,
            "total_count": len(group),
            "overall_avg_tat": round(group['tat'].mean(), 1),
            "urgent_avg_tat": round(urgent_grp['tat'].mean(), 1) if len(urgent_grp) else 0,
            "routine_avg_tat": round(routine_grp['tat'].mean(), 1) if len(routine_grp) else 0,
            "max_tat": round(group['tat'].max(), 1),
            "target_sla_minutes": 60 if t_type in ['ECG', 'Urine test'] else 90,
            "sla_breach_rate": round((group['tat'] > (60 if t_type in ['ECG', 'Urine test'] else 90)).mean() * 100, 1)
        })

    # Export complete results to JSON for web prototype
    results_payload = {
        "metadata": {
            "title": "CareFlow Radar - Hospital Flow ML Engine",
            "evaluated_at": datetime.now().isoformat(),
            "time_aware_split_date": str(split_date),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "total_visits": len(df)
        },
        "model_performance": {
            "baseline_global_mean": {"mae": round(base_mae, 2), "rmse": round(base_rmse, 2), "r2": round(base_r2, 4)},
            "baseline_triage_only": {"mae": round(triage_mae, 2), "rmse": round(triage_rmse, 2), "r2": round(triage_r2, 4)},
            "ridge_regressor_from_scratch": {"mae": round(r_mae, 2), "rmse": round(r_rmse, 2), "r2": round(r_r2, 4)},
            "gbdt_ensemble_from_scratch": {"mae": round(g_mae, 2), "rmse": round(g_rmse, 2), "r2": round(g_r2, 4)},
            "delay_classifier": {
                "accuracy": round(accuracy * 100, 1),
                "precision": round(precision * 100, 1),
                "recall": round(recall * 100, 1),
                "f1_score": round(f1, 3),
                "roc_auc": round(auc_est, 3)
            }
        },
        "feature_attributions": [
            {"feature": feat, "weight": round(w, 3), "importance_pct": round(imp * 100, 1)}
            for feat, w, imp in top_features
        ],
        "department_kpis": dept_kpis,
        "investigation_queue": investigation_queue[:18],
        "hourly_patterns": hourly_patterns,
        "diagnostic_tat_monitor": tat_summary
    }

    with open("model_results.json", "w") as f:
        json.dump(results_payload, f, indent=2)

    print("  Successfully generated 'model_results.json'!")
    return results_payload

if __name__ == "__main__":
    df, departments, patients, encounters, tests = load_and_preprocess_data()
    results = train_and_evaluate(df)
