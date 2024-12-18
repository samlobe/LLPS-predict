#%%
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time

# Set model name
which_model = 'esm3-small-2024-08'  # Replace with the desired model

# Load embeddings
driver_embeddings = pd.read_csv(f'features/driver_embeddings_{which_model}.csv', index_col=0)
non_driver_embeddings = pd.read_csv(f'features/non_driver_embeddings_{which_model}.csv', index_col=0)

# Create labels
X_drivers = np.array(driver_embeddings)
y_drivers = np.ones(len(X_drivers))  # Label for drivers: 1

X_non_drivers_all = np.array(non_driver_embeddings)
y_non_drivers_all = np.zeros(len(X_non_drivers_all))  # Label for non-drivers: 0

C1_values = [0.01, 0.05, 0.1, 0.5]
C2_values = [1, 0.1, 0.01, 0.001, 0.0001]

# Function to evaluate params with CV
def evaluate_params(C1, C2, X_train, y_train, skf):
    """Evaluate a given (C1, C2) parameter combination via CV."""
    scores = []
    for train_index, val_index in skf.split(X_train, y_train):
        # Get the data splits
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Standardize the data
        scaler = StandardScaler()
        X_train_fold_scaled = scaler.fit_transform(X_train_fold)
        X_val_fold_scaled = scaler.transform(X_val_fold)

        # L1 logistic regression for feature selection
        lr_l1 = LogisticRegression(penalty='l1', C=C1, solver='liblinear', max_iter=10000)
        lr_l1.fit(X_train_fold_scaled, y_train_fold)

        # Select non-zero coefficients
        non_zero_indices = np.where(lr_l1.coef_[0] != 0)[0]
        if len(non_zero_indices) == 0:
            # If no features selected, skip this combination
            return None

        # Select features
        X_train_fold_selected = X_train_fold_scaled[:, non_zero_indices]
        X_val_fold_selected = X_val_fold_scaled[:, non_zero_indices]

        # L2 logistic regression
        lr_l2 = LogisticRegression(penalty='l2', C=C2, solver='lbfgs', max_iter=10000, class_weight='balanced')
        lr_l2.fit(X_train_fold_selected, y_train_fold)

        # Predict probabilities on validation fold and store scores
        y_val_pred = lr_l2.predict_proba(X_val_fold_selected)[:, 1]
        score = roc_auc_score(y_val_fold, y_val_pred)
        scores.append(score)

    if len(scores) == 0:
        return None
    return np.mean(scores)

# Number of train/val splits to run
num_splits = 30
auc_scores = []

for split_num in tqdm(range(num_splits)):
    # Randomly set aside 20% of each dataset for validation
    X_drivers_train, X_drivers_val, y_drivers_train, y_drivers_val = train_test_split(
        X_drivers, y_drivers, test_size=0.2, stratify=y_drivers, random_state=42 + split_num)
    X_non_drivers_train, X_non_drivers_val, y_non_drivers_train, y_non_drivers_val = train_test_split(
        X_non_drivers_all, y_non_drivers_all, test_size=0.2, stratify=y_non_drivers_all, random_state=42 + split_num)

    X_train = np.concatenate([X_drivers_train, X_non_drivers_train], axis=0)
    y_train = np.concatenate([y_drivers_train, y_non_drivers_train], axis=0)
    X_val = np.concatenate([X_drivers_val, X_non_drivers_val], axis=0)
    y_val = np.concatenate([y_drivers_val, y_non_drivers_val], axis=0)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    param_combinations = [(C1, C2) for C1 in C1_values for C2 in C2_values]

    # Evaluate all parameter combinations in parallel
    results = Parallel(n_jobs=20, verbose=0)(
        delayed(evaluate_params)(C1, C2, X_train, y_train, skf) for (C1, C2) in param_combinations
    )

    best_score = -1
    best_C1 = None
    best_C2 = None

    for (C1, C2), score in zip(param_combinations, results):
        if score is not None and score > best_score:
            best_score = score
            best_C1 = C1
            best_C2 = C2

    # Retrain with best parameters
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # L1 logistic regression on whole training set
    lr_l1 = LogisticRegression(penalty='l1', C=best_C1, solver='liblinear', max_iter=10000)
    lr_l1.fit(X_train_scaled, y_train)

    # Get non-zero indices
    non_zero_indices = np.where(lr_l1.coef_[0] != 0)[0]

    # Select features
    X_train_selected = X_train_scaled[:, non_zero_indices]

    # L2 logistic regression
    lr_l2 = LogisticRegression(penalty='l2', C=best_C2, solver='lbfgs', max_iter=10000, class_weight='balanced')
    lr_l2.fit(X_train_selected, y_train)

    X_val_scaled = scaler.transform(X_val)
    X_val_selected = X_val_scaled[:, non_zero_indices]

    y_val_pred = lr_l2.predict_proba(X_val_selected)[:, 1]

    roc_auc = roc_auc_score(y_val, y_val_pred)
    auc_scores.append(roc_auc)

    # Append the ROC AUC to a txt file for this model
    with open(f'{which_model}_auc_scores.txt', 'a') as f:
        f.write(f"Split {split_num+1}, AUC: {roc_auc}\n")

# Calculate the 16th, 50th, and 84th percentiles
p16 = np.percentile(auc_scores, 16)
p50 = np.percentile(auc_scores, 50)
p84 = np.percentile(auc_scores, 84)

# Append to esm_performances.csv
# Ensure that esm_performances.csv exists or create it if not
# We will write: model_name, 16th percentile, median, 84th percentile
esm_performances_path = 'esm_model_performances.csv'

header_needed = False
try:
    # Check if file exists
    with open(esm_performances_path, 'r') as f:
        pass
except FileNotFoundError:
    header_needed = True

with open(esm_performances_path, 'a') as f:
    if header_needed:
        f.write("model,16th_percentile,50th_percentile,84th_percentile\n")
    f.write(f"{which_model},{p16:.4f},{p50:.4f},{p84:.4f}\n")

print("All done. AUC scores saved to txt and percentiles appended to esm_performances.csv.")

# %%
