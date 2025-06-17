# This tutorial demonstrates how to train and evaluate a screening model based on XGBoost,
# using Bayesian Optimization and stratified k-fold cross-validation.
# Each fold's best model is saved, and validation and external test metrics are recorded.
# If you encounter any issues, feel free to open an issue on GitHub.

import pandas as pd
import numpy as np
import os
import time
import joblib
import warnings
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score



warnings.filterwarnings('ignore')



# ==============================
# Evaluation function
# ==============================

def evaluation(model, X_test, y_true):
    """
    Evaluate classification performance of a scikit-learn compatible model.

    Parameters
    ----------
    model : trained classification model
    X_test : np.ndarray or pd.DataFrame
        Test features
    y_true : np.ndarray or pd.Series
        True labels for X_test

    Returns
    -------
    accuracy : float
    sensitivity : float (recall for positive class)
    specificity : float (recall for negative class)
    precision : float
    f1 : float
    auc : float
    """
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, proba)

    return accuracy, sensitivity, specificity, precision, f1, auc



# ==============================
# Configuration section
# ==============================

DATA_PATH = 'your_data_path.xlsx'  # Path to the Excel dataset
EXTERNAL_TEST_SIZE = 500  # Number of external test samples (adjust as needed)
N_SPLITS = 5  # Stratified K-Fold cross-validation
INIT_POINTS = 50  # Initial points for Bayesian Optimization
N_ITER = 100  # Optimization iterations
MODEL_DIR = 'model'  # Directory to save models
RESULTS_DIR = 'Results'  # Directory to save evaluation metrics

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Hyperparameter search space for XGBoost
param_space_xgb = {
    'max_depth': (3, 8),
    'learning_rate': (0.01, 0.5),
    'n_estimators': (50, 500),
    'gamma': (0, 1),
    'subsample': (0.5, 1),
    'colsample_bytree': (0.5, 1),
    'reg_alpha': (0, 1),
    'reg_lambda': (0, 1)
}

# ==============================
# Begin script
# ==============================

start_time = time.time()

# Load dataset (assumes label is last column and first column is an ID or index)
data = pd.read_excel(DATA_PATH)
X = np.array(data.iloc[:, 1:-1])
y = np.array(data.iloc[:, -1])

# Split into training + CV set and an external test set
X_train_data, X_extral_test, y_train_data, y_extral_test = train_test_split(
    X, y, test_size=EXTERNAL_TEST_SIZE, random_state=42, stratify=y
)

# Prepare result containers
metrics_columns = ['accuracy', 'sensitivity', 'specificity', 'roc_auc']
xgb_metrics_df = pd.DataFrame(columns=metrics_columns)
xgb_metrics_df_test = pd.DataFrame(columns=metrics_columns)

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n========== Fold {fold} ==========")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # -----------------------------------
    # Define objective function for Bayesian Optimization
    # -----------------------------------
    def xgb_cv(**params):
        model = XGBClassifier(
            max_depth=int(params['max_depth']),
            learning_rate=params['learning_rate'],
            n_estimators=int(params['n_estimators']),
            gamma=params['gamma'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        _, _, specificity, _, _, _ = evaluation(model, X_val, y_val)
        return specificity  # Use specificity as optimization target

    # -----------------------------------
    # Run Bayesian Optimization
    # -----------------------------------
    optimizer = BayesianOptimization(
        f=xgb_cv,
        pbounds=param_space_xgb,
        random_state=42,
        verbose=0
    )
    optimizer.maximize(init_points=INIT_POINTS, n_iter=N_ITER)
    best_params = optimizer.max['params']

    # -----------------------------------
    # Bayesian Optimization only returns the best hyperparameters, not the best model.
    # Therefore, we retrain the model using these optimal parameters to obtain
    # the actual best-performing model for evaluation and saving.
    # -----------------------------------
    best_model = XGBClassifier(
        max_depth=int(best_params['max_depth']),
        learning_rate=best_params['learning_rate'],
        n_estimators=int(best_params['n_estimators']),
        gamma=best_params['gamma'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        reg_alpha=best_params['reg_alpha'],
        reg_lambda=best_params['reg_lambda'],
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    best_model.fit(X_train, y_train)

    # -----------------------------------
    # Evaluate model
    # -----------------------------------
    acc, sen, spe, _, _, auc = evaluation(best_model, X_val, y_val)
    xgb_metrics_df.loc[len(xgb_metrics_df)] = [acc, sen, spe, auc]

    acc, sen, spe, _, _, auc = evaluation(best_model, X_extral_test, y_extral_test)
    xgb_metrics_df_test.loc[len(xgb_metrics_df_test)] = [acc, sen, spe, auc]

    # -----------------------------------
    # Save model
    # -----------------------------------
    joblib.dump(best_model, f'{MODEL_DIR}/xgb_fold_{fold}.joblib')

# ==============================
# Save results
# ==============================

xgb_metrics_df.to_excel(f'{RESULTS_DIR}/xgb_metrics_df.xlsx', index=False)
xgb_metrics_df_test.to_excel(f'{RESULTS_DIR}/xgb_metrics_df_test.xlsx', index=False)

# ==============================
# Print summary statistics
# ==============================

print("\nTrain metrics summary:")
print(xgb_metrics_df.mean(), xgb_metrics_df.std(), xgb_metrics_df.var())

print("\nTest metrics summary:")
print(xgb_metrics_df_test.mean(), xgb_metrics_df_test.std(), xgb_metrics_df_test.var())

# ==============================
# Execution time
# ==============================

elapsed = (time.time() - start_time) / 3600
print(f"\nTotal execution time: {elapsed:.2f} hours")
