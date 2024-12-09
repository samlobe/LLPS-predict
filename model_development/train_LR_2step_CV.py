#%%
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from time import time
from tqdm import tqdm

# Load embeddings
driver_embeddings = joblib.load('features/driver_embeddings.pkl')
non_driver_embeddings = joblib.load('features/non_driver_embeddings.pkl')

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

# Define parameter grids
C1_values = [0.01, 0.05, 0.1, 0.5]
C2_values = [1, 0.1, 0.01, 0.001, 0.0001]

best_score = 0
best_C1 = None
best_C2 = None

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for C1 in tqdm(C1_values):
    for C2 in C2_values:
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
                continue

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

        # Compute average performance
        if len(scores) == 0:
            continue  # skip if no scores (e.g., when no features selected)

        avg_score = np.mean(scores)
        print(f"C1={C1}, C2={C2}, Avg ROC AUC: {avg_score:.4f}")

        # Update best parameters if performance is better
        if avg_score > best_score:
            best_score = avg_score
            best_C1 = C1
            best_C2 = C2

# After looping, we have best_C1 and best_C2
print(f"Best C1: {best_C1}, Best C2: {best_C2}, Best ROC AUC: {best_score:.4f}")

# Retrain on entire training set with best parameters
# Standardize the training data
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

# Apply to validation set
X_val = np.concatenate([X_drivers_val, X_non_drivers_val], axis=0)
y_val = np.concatenate([y_drivers_val, y_non_drivers_val], axis=0)
X_val_scaled = scaler.transform(X_val)
X_val_selected = X_val_scaled[:, non_zero_indices]

# Predict probabilities
y_val_pred = lr_l2.predict_proba(X_val_selected)[:, 1]

# Evaluate performance
roc_auc = roc_auc_score(y_val, y_val_pred)
average_precision = average_precision_score(y_val, y_val_pred)
print(f"Validation ROC AUC: {roc_auc:.4f}")
print(f"Validation Average Precision: {average_precision:.4f}")

#%%
# Plot the ROC curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_val, y_val_pred)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve, AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curve: LLPS Classifier', fontsize=16)
plt.xticks(fontsize=14); plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.xlim([0, 1]); plt.ylim([0, 1.05])
plt.show()

# plot the precision-recall curve
from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(y_val, y_val_pred)
plt.figure()
plt.plot(recall, precision, label=f'Average Precision = {average_precision:.2f}')
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Precision-Recall Curve: LLPS Classifier', fontsize=16)
plt.xticks(fontsize=14); plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.xlim([0, 1]); plt.ylim([0, 1.00])
plt.show()


#%%
# Use the previously selected features for final training

# Concatenate training and validation data
X_full = np.concatenate([X_train, X_val], axis=0)
y_full = np.concatenate([y_train, y_val], axis=0)

# Standardize the full dataset using the scaler fitted on the training data
X_full_scaled = scaler.transform(X_full)  # Use the scaler fitted earlier on X_train

# Select the same features as before
X_full_selected = X_full_scaled[:, non_zero_indices]

# Retrain the final model on the full dataset using the selected features
lr_final = LogisticRegression(penalty='l2', C=best_C2, solver='lbfgs', max_iter=10000, class_weight='balanced')
lr_final.fit(X_full_selected, y_full)

# Save the scaler, feature indices, and the trained model using joblib
model_dict = {
    'scaler': scaler,               # The scaler fitted on the training data
    'feature_indices': non_zero_indices,  # Indices of selected features
    'model': lr_final               # The trained logistic regression model
}

joblib.dump(model_dict, 'LLPS_model_latest.joblib')
print("Final model saved as 'LLPS_model_latest.joblib'")

#%%
