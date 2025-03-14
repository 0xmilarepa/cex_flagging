import numpy as np
import pandas as pd
from typing import Tuple

def find_addresses_in_df(df, columns, seed_addr=None, cross_reference=True):

    """
    This function extracts unique addresses from specified columns in a DataFrame and optionally
    cross-references them with a provided list of addresses.

    Parameters:
        df (pd.DataFrame): DataFrame containing blockchain transaction data.
        columns (list): List of column names in the DataFrame that contain wallet addresses.
        seed_addr (set or list, optional): Set or list of addresses to cross-reference with the DataFrame.
                                           Defaults to None.
        cross_reference (bool, optional): If True, performs cross-referencing with `seed_addr`.
                                          If False, returns all unique addresses found in the specified columns.
                                          Defaults to True.

    Returns:
        set: A set of addresses. If `cross_reference` is True, returns addresses present in both
             `seed_addr` and the specified columns. If `cross_reference` is False, returns all
             unique addresses found in the specified columns.
    """

    addresses_found = set()
    for col in columns:
        if col in df.columns:
            addresses_found.update(df[col].dropna().unique())

    if cross_reference and seed_addr is not None:
        return seed_addr.intersection(addresses_found)
    else:
        return addresses_found



def prediction_df(
    y_pred_proba: np.ndarray,
    addresses: np.ndarray,
    threshold: float = 0.7
) -> pd.DataFrame:

    """
    This function generates a DataFrame containing predictions, probabilities, and flagged addresses based on a custom threshold.

    Args:
        y_pred_proba (np.ndarray): A 2D NumPy array of shape (n_samples, 2) containing predicted probabilities
                                   for each class (e.g., [probability_class_0, probability_class_1]).
        addresses (np.ndarray): A 1D NumPy array of shape (n_samples,) containing the addresses corresponding
                                to the predictions.
        threshold (float, optional): Probability threshold for classifying an address as the positive class (default: 0.7).

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
                      - 'address': The blockchain addresses.
                      - 'prediction': Binary classification (0 or 1) based on the threshold.
                      - 'probability': Confidence level of the prediction (probability of the positive class).
                      - 'flagged': Boolean indicating whether the address is flagged as a CEX (prediction == 1).

    """
    # Ensuring input shapes are valid
    if y_pred_proba.shape[1] != 2:
        raise ValueError("y_pred_proba must have shape (n_samples, 2).")
    if len(addresses) != y_pred_proba.shape[0]:
        raise ValueError("Length of addresses must match the number of samples in y_pred_proba.")

    # Extracting probabilities for the positive class (class 1)
    probabilities = y_pred_proba[:, 1]

    # Apply threshold to generate binary predictions
    predictions = (probabilities >= threshold).astype(int)

    # Create DataFrame
    results_df = pd.DataFrame({
        'address': addresses,
        'prediction': predictions,
        'probability': probabilities,
    })

    return results_df
