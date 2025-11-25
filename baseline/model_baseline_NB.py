import logging
import os
import numpy as np
import json
from sklearn.naive_bayes import MultinomialNB
from utils.preprocess import TFIDFLogRegPreprocessor # We reuse the TF-IDF preprocessor
from utils.evaluate import evaluate_and_plot

logger = logging.getLogger(__name__)

def train_nb_model(X_train, y_train, X_val, y_val, output_dir: str):
    """
    Trains and evaluates the Multinomial Naive Bayes classifier (Baseline D).
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
    logger.info("Starting Naive Bayes preprocessing (TF-IDF)...")
    
    # 1. Preprocessing (Fit on Training Data only)
    preprocessor = TFIDFLogRegPreprocessor()
    preprocessor.fit(X_train)
    
    X_train_vec = preprocessor.transform(X_train)
    X_val_vec = preprocessor.transform(X_val)
    
    # 2. Model Training
    # Naive Bayes is generally fast and works well with sparse TF-IDF data
    model = MultinomialNB()
    logger.info("Training Naive Bayes model...")
    model.fit(X_train_vec, y_train)
    
    # 3. Prediction
    logger.info("Making predictions on validation set...")
    y_pred = model.predict(X_val_vec)
    
    # 4. Evaluation and Plotting
    num_labels = len(np.unique(np.concatenate((y_train, y_val))))
    final_macro_f1 = evaluate_and_plot(
        y_val, y_pred, output_dir, num_labels, model_name='nb_final'
    )
    
    # Save final results to JSON
    results_path = os.path.join(output_dir, 'nb_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'model': 'Naive Bayes (TF-IDF)',
            'macro_f1': final_macro_f1,
            'features': 'TF-IDF',
        }, f, indent=4)
        
    logger.info(f"Final results saved to {results_path}")
    return final_macro_f1