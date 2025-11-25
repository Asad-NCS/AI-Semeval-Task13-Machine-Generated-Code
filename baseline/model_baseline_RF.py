import logging
import os
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from utils.preprocess import TFIDFLogRegPreprocessor
from utils.evaluate import evaluate_and_plot

logger = logging.getLogger(__name__)

def train_rf_model(X_train, y_train, X_val, y_val, output_dir: str):
    """
    Trains and evaluates the Random Forest classifier (Baseline E).
    This baseline uses TF-IDF features.
    
    Args:
        X_train (list): List of training code snippets.
        y_train (np.array): Array of training labels.
        X_val (list): List of validation code snippets.
        y_val (np.array): Array of validation labels.
        output_dir (str): Directory to save plots and results.
    
    Returns:
        float: The final macro F1 score on the validation set.
    """
    logger.info("Starting Random Forest preprocessing (TF-IDF)...")
    
    # 1. Preprocessing (Fit on Training Data only)
    preprocessor = TFIDFLogRegPreprocessor()
    preprocessor.fit(X_train)
    
    X_train_vec = preprocessor.transform(X_train)
    X_val_vec = preprocessor.transform(X_val)
    
    # 2. Model Training
    # Random Forest is an ensemble method, generally robust.
    # Using sensible defaults for n_estimators and random_state for reproducibility.
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    logger.info("Training Random Forest model (100 estimators)...")
    model.fit(X_train_vec, y_train)
    
    # 3. Prediction
    logger.info("Making predictions on validation set...")
    y_pred = model.predict(X_val_vec)
    
    # 4. Evaluation and Plotting
    num_labels = len(np.unique(np.concatenate((y_train, y_val))))
    final_macro_f1 = evaluate_and_plot(
        y_val, y_pred, output_dir, num_labels, model_name='rf_final'
    )
    
    # Save final results to JSON
    results_path = os.path.join(output_dir, 'rf_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'model': 'Random Forest (TF-IDF)',
            'macro_f1': final_macro_f1,
            'features': 'TF-IDF',
            'n_estimators': 100
        }, f, indent=4)
        
    logger.info(f"Final results saved to {results_path}")
    return final_macro_f1