import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import os
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# --- Supervised Models ---

def initialize_supervised_model(model_type):
    """
    This function initializes a supervised model based on the specified type.

    Parameters:
        model_type (str): Type of model to initialize. Options: 'random_forest', 'logistic_regression'.

    Returns:
        model: An untrained model.
    """
    if model_type == 'random_forest':
        return RandomForestClassifier(random_state=42, class_weight='balanced')
    elif model_type == 'logistic_regression':
        return LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    else:
        raise ValueError(f"Unknown supervised model type: {model_type}")


def compile_supervised_model(model, X_train, y_train):

    """
    This function performs hyperparameter tuning using GridSearchCV for supervised models.

    Parameters:
        model: The untrained model instance
        X_train: Training features
        y_train: Training labels

    Returns:
        model: The best model after hyperparameter tuning
    """

    if isinstance(model, RandomForestClassifier):
        param_grid = {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [None, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }
        scoring = 'f1'

    elif isinstance(model, LogisticRegression):
        param_grid = {
            'model__C': [0.01, 0.1, 1, 10, 100],
            'model__penalty': ['l1', 'l2'],
            'model__solver': ['liblinear']
        }
        scoring = 'f1'

    else:
        raise ValueError(f"Unsupported supervised model type: {type(model)}")

    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', model)
    ])

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=scoring, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Best Parameters for {type(model).__name__}:", grid_search.best_params_)
    return grid_search.best_estimator_

def evaluate_supervised_model(model, X, y, set_name="Validation"):

    """
    This function evaluates a supervised model on a given dataset.

    Parameters:
        model: The trained model
        X: Features of the dataset
        y: Labels of the dataset
        set_name: Name of the dataset (e.g., "Validation", "Test")
    """
    y_pred = model.predict(X)

    print(f"\nEvaluation metrics for {set_name} set:")
    print("Accuracy:", accuracy_score(y, y_pred))
    print("F1-Score (Weighted):", f1_score(y, y_pred, average='weighted'))
    print("F1-Score (Macro):", f1_score(y, y_pred, average='macro'))

    if hasattr(model, 'predict_proba'):
        if len(np.unique(y)) == 2:
            print("ROC-AUC Score:", roc_auc_score(y, model.predict_proba(X)[:, 1]))
        else:
            print("ROC-AUC Score (OvR):", roc_auc_score(y, model.predict_proba(X), multi_class='ovr', average='weighted'))

    print("Classification Report:\n", classification_report(y, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))

    if hasattr(model, 'feature_importances_'):
        if isinstance(X, pd.DataFrame):
            feature_importances = pd.Series(model.feature_importances_, index=X.columns)
        else:
            feature_importances = pd.Series(model.feature_importances_, index=[f'feature_{i}' for i in range(X.shape[1])])
        print("Feature Importances:\n", feature_importances.sort_values(ascending=False))


# --- Unsupervised Models ---


def iso_forest(df, feature_columns, y, results_folder='results'):
    '''
    This function trains an Isolation Forest model for anomaly detection, and saves the model with its results.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the dataset.
        feature_columns (list): List of column names to be used as features.
        y (pd.Series): True labels for evaluation
        results_folder (str, optional): Path to the folder where results will be saved. Default is 'results'.

    Returns:
        pd.DataFrame: DataFrame containing addresses and anomaly flags (is_cex).
        tuple: Evaluation metrics (precision, recall, f1, roc_auc, confusion_matrix)
    '''
    # Ensure the results folder exists
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_samples': ['auto', 100, 200],
        'contamination': [0.01, 0.03, 0.05, 0.1],
        'max_features': [0.5, 0.75, 1.0]
    }

    # Initialize the Isolation Forest model
    iso_forest = IsolationForest()

    # Prepare the feature set
    X_iso = df[feature_columns]
    y = df[y]

    # Grid Search (using the initial model)
    grid_search = GridSearchCV(iso_forest, param_grid, scoring='roc_auc', cv=5, n_jobs=-1)
    grid_search.fit(X_iso, y)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Fit the best model and predict anomalies
    iso_pred = best_model.fit_predict(X_iso)

    # Create a DataFrame to store the results
    if 'address' not in df.columns:
        raise ValueError("DataFrame must contain an 'address' column")

    final_flags = pd.DataFrame({
        'address': df['address'],
        'is_cex': (iso_pred == -1).astype(int)  # Convert -1 to 1 for anomalies
    })

    # Convert predictions to match y's format (0,1) instead of (-1,1)
    y_pred_binary = (iso_pred == -1).astype(int)

    # Evaluation metrics
    cm = confusion_matrix(y, y_pred_binary)
    precision = precision_score(y, y_pred_binary)
    recall = recall_score(y, y_pred_binary)
    f1 = f1_score(y, y_pred_binary)
    roc_auc = roc_auc_score(y, y_pred_binary)

    # Print results
    print("Anomaly-based CEX Flags:", final_flags['is_cex'].sum())
    print("Best Parameters:", grid_search.best_params_)
    print("ROC-AUC Score:", roc_auc)
    print("Confusion Matrix:\n", cm)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)

    # Define the file paths
    model_filename = os.path.join(results_folder, 'isolation_forest_model.pkl')
    results_filename = os.path.join(results_folder, 'isolation_forest_predictions.csv')

    # Save the trained model
    joblib.dump(best_model, model_filename)
    print(f"Model saved to {model_filename}")

    # Save the results DataFrame to a CSV file
    final_flags.to_csv(results_filename, index=False)
    print(f"Results saved to {results_filename}")

    return final_flags, (precision, recall, f1, roc_auc, cm)
