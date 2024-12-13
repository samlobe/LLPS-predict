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
from time import time

which_model = 'esm3-small-2024-08'  # Replace with the desired model
# Load embeddings
driver_embeddings = pd.read_csv(f'features/driver_embeddings_{which_model}.csv', index_col=0)
non_driver_embeddings = pd.read_csv(f'features/non_driver_embeddings_{which_model}.csv', index_col=0)

# Create labels
X_drivers = np.array(driver_embeddings)
y_drivers = np.ones(len(X_drivers))  # Label for drivers: 1

X_non_drivers_all = np.array(non_driver_embeddings)
y_non_drivers_all = np.zeros(len(X_non_drivers_all))  # Label for non-drivers: 0

# Randomly set aside 20% of each dataset for validation
X_drivers_train, X_drivers_val, y_drivers_train, y_drivers_val = train_test_split(
    X_drivers, y_drivers, test_size=0.2, stratify=y_drivers, random_state=42)
X_non_drivers_train, X_non_drivers_val, y_non_drivers_train, y_non_drivers_val = train_test_split(
    X_non_drivers_all, y_non_drivers_all, test_size=0.2, stratify=y_non_drivers_all, random_state=42)

X_train = np.concatenate([X_drivers_train, X_non_drivers_train], axis=0)
y_train = np.concatenate([y_drivers_train, y_non_drivers_train], axis=0)
X_val = np.concatenate([X_drivers_val, X_non_drivers_val], axis=0)
y_val = np.concatenate([y_drivers_val, y_non_drivers_val], axis=0)

C1_values = [0.01, 0.05, 0.1, 0.5]
C2_values = [1, 0.1, 0.01, 0.001, 0.0001]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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

param_combinations = [(C1, C2) for C1 in C1_values for C2 in C2_values]

print(f"Evaluating {len(param_combinations)} parameter combinations in parallel...")
results = Parallel(n_jobs=20, verbose=10)(
    delayed(evaluate_params)(C1, C2, X_train, y_train, skf) for (C1, C2) in param_combinations
)

best_score = 0
best_C1 = None
best_C2 = None

for (C1, C2), score in zip(param_combinations, results):
    if score is not None:
        print(f"C1={C1}, C2={C2}, Avg ROC AUC: {score:.4f}")
        if score > best_score:
            best_score = score
            best_C1 = C1
            best_C2 = C2

print(f"Best C1: {best_C1}, Best C2: {best_C2}, Best ROC AUC: {best_score:.4f}")

########################################
# Step 4: Retrain with best parameters and evaluate on consistent validation set
########################################
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# L1 logistic regression on whole training set
lr_l1 = LogisticRegression(penalty='l1', C=best_C1, solver='liblinear', max_iter=10000)
lr_l1.fit(X_train_scaled, y_train)

# Get non-zero indices
non_zero_indices = np.where(lr_l1.coef_[0] != 0)[0]
print(f"Number of non-zero coefficients: {len(non_zero_indices)}")

# Select features
X_train_selected = X_train_scaled[:, non_zero_indices]

# L2 logistic regression
lr_l2 = LogisticRegression(penalty='l2', C=best_C2, solver='lbfgs', max_iter=10000, class_weight='balanced')
lr_l2.fit(X_train_selected, y_train)

X_val_scaled = scaler.transform(X_val)
X_val_selected = X_val_scaled[:, non_zero_indices]

y_val_pred = lr_l2.predict_proba(X_val_selected)[:, 1]

roc_auc = roc_auc_score(y_val, y_val_pred)
average_precision = average_precision_score(y_val, y_val_pred)
print(f"Validation ROC AUC: {roc_auc:.4f}")
print(f"Validation Average Precision: {average_precision:.4f}")

# Plot the ROC curve
fpr, tpr, _ = roc_curve(y_val, y_val_pred)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve, AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curve: LLPS Classifier', fontsize=16)
plt.legend(fontsize=14)
plt.xlim([0, 1]); plt.ylim([0, 1.05])
plt.show()

# Save ROC curve data
df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
df.to_csv(f'roc_curves/{which_model}.csv', index=False)

# Append AUC to file
with open('roc_curves/aucs.txt', 'a') as f:
    f.write(f'{which_model}, {roc_auc}\n')

# %%
