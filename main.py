from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from models import *

def main_sup(df, feature_columns, target_column, results_folder='results', model_type=None, sampling_strategy=0.5, threshold=0.7):
    '''
    This function trains and evaluates two supervised machine learning models (Random Forest, Logistic Regression).

    Parameters:
        df (pd.DataFrame): The full DataFrame with features, target, and addresses
        feature_columns (list): List of column names to be used as features.
        target_column (str): Name of the column to be used as the target variable.
        results_folder (str): Path to the folder where results will be saved. Default is 'results'.
        model_type (str, optional): Type of model to train. Options: 'random_forest', 'logistic_regression'.
                                   If None, all models will be trained.
        sampling_strategy (float): The desired ratio of minority class to majority class after resampling.
                                   Default is 0.2.
        threshold (float): Probability threshold for classification. Default is 0.7.

    Returns:
        None: Saves trained models and predictions to the specified results folder.
    '''
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Extract features, target, and addresses from DataFrame
    X = df[feature_columns]
    y = df[target_column]
    addresses = df['address']

    if X.empty or y.empty:
        raise ValueError("Feature or target data is empty.")
    if X.isnull().any().any() or y.isnull().any():
        raise ValueError("Missing values detected.")

    # 1. Train-Test Split
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    # Preserve test addresses for later use
    test_addresses = addresses.loc[X_test.index]

    # 2. Apply SMOTETomek to the training data
    smotetomek = SMOTETomek(sampling_strategy=sampling_strategy, random_state=42)
    X_train_resampled, y_train_resampled = smotetomek.fit_resample(X_train, y_train)

    print(f"Class distribution before SMOTETomek: {Counter(y_train)}")
    print(f"Class distribution after SMOTETomek: {Counter(y_train_resampled)}")

    if model_type is None:
        models_to_train = ['random_forest', 'logistic_regression']
    else:
        if model_type not in ['random_forest', 'logistic_regression']:
            raise ValueError(f"Invalid model_type: {model_type}. Choose from 'random_forest', 'logistic_regression'.")
        models_to_train = [model_type]

    for model_type in models_to_train:
        print(f"\nTraining {model_type} ✅")
        model = initialize_supervised_model(model_type)
        best_model = compile_supervised_model(model, X_train_resampled, y_train_resampled)
        evaluate_supervised_model(best_model, X_val, y_val, set_name="Validation")
        evaluate_supervised_model(best_model, X_test, y_test, set_name="Test")

        # Save the trained model
        model_filename = os.path.join(results_folder, f'{model_type}_model.pkl')
        joblib.dump(best_model, model_filename)

        # Get predictions with threshold for test set
        X_test_scaled = best_model.named_steps['scaler'].transform(X_test)
        y_pred_proba = best_model.named_steps['model'].predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_pred_proba > threshold).astype(int)

        # 3. Save test predictions with addresses as test_results
        test_results = pd.DataFrame({
            'address': test_addresses,
            'prediction': y_pred,
            'probability': y_pred_proba
        })
        test_results_filename = os.path.join(results_folder, f'{model_type}_test_predictions.csv')
        test_results.to_csv(test_results_filename, index=False)

        print(f"\nTest set predictions for {model_type}:")
        print(test_results.head())

        # 4. Predict on all data
        X_all_scaled = best_model.named_steps['scaler'].transform(X)
        all_pred_proba = best_model.named_steps['model'].predict_proba(X_all_scaled)[:, 1]
        all_pred = (all_pred_proba > threshold).astype(int)

        final_flags = pd.DataFrame({
            'address': addresses,
            'is_cex': all_pred
        })
        full_results_filename = os.path.join(results_folder, f'{model_type}_full_predictions.csv')
        final_flags.to_csv(full_results_filename, index=False)

        print(f"\nUnique addresses flagged as CEX by {model_type}: {final_flags['is_cex'].sum()}")
        print(f"\nFirst few flagged addresses for {model_type}:")
        print(final_flags.head())
