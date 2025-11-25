# baseline/model_baseline_A.py

import os
import json
import logging
from sklearn.linear_model import LogisticRegression
from utils.preprocess import TFIDFLogRegPreprocessor
from utils.evaluate import evaluate_and_plot

logger = logging.getLogger(__name__)

class TFIDFLogRegModel:
    """
    Baseline A: TF-IDF + Logistic Regression Model
    Includes:
        - Preprocessing
        - Training
        - Evaluation
    """

    def __init__(self, max_features=50000, ngram_range=(1,2), token_pattern=r"\b\w+\b"):
        self.preprocessor = TFIDFLogRegPreprocessor(
            max_features=max_features,
            ngram_range=ngram_range,
            token_pattern=token_pattern
        )
        self.model = LogisticRegression(max_iter=2000, n_jobs=-1)
        self.fitted = False

    def fit(self, X_train, y_train):
        """
        Fit TF-IDF vectorizer on training data and train Logistic Regression.
        """
        logger.info("[Baseline A] Fitting TF-IDF vectorizer on training data...")
        X_train_tfidf = self.preprocessor.fit_transform(X_train)
        logger.info("[Baseline A] Training Logistic Regression model...")
        self.model.fit(X_train_tfidf, y_train)
        self.fitted = True
        logger.info("[Baseline A] Training complete.")

    def predict(self, X):
        """
        Predict labels for given texts.
        """
        if not self.fitted:
            raise ValueError("[Baseline A] Model not trained. Call fit() first.")
        X_tfidf = self.preprocessor.transform(X)
        return self.model.predict(X_tfidf)

    def evaluate(self, X_val, y_val, output_dir, model_name='baselineA'):
        """
        Evaluate model using validation data and save metrics & confusion matrix.
        Returns Macro F1.
        """
        y_pred = self.predict(X_val)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        macro_f1 = evaluate_and_plot(
            y_true=y_val,
            y_pred=y_pred,
            output_dir=output_dir,
            num_labels=len(set(y_val)),
            model_name=model_name
        )
        logger.info(f"[Baseline A] Macro F1 on validation set: {macro_f1:.4f}")

        return macro_f1


def train_tfidf_logreg_model(X_train, y_train, X_val, y_val, output_dir):
    """
    Convenience function: trains and evaluates Baseline A.
    Returns Macro F1.
    """
    model = TFIDFLogRegModel()
    model.fit(X_train, y_train)
    macro_f1 = model.evaluate(X_val, y_val, output_dir)
    results_path = os.path.join(output_dir, 'baselineA_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'model': 'TFIDF + LogisticRegression',
            'macro_f1': macro_f1,
            'max_features': model.preprocessor.max_features,
            'ngram_range': model.preprocessor.ngram_range,
            'vocab_size': len(model.preprocessor.vectorizer.vocabulary_)
        }, f, indent=4)
    
    logger.info(f"Final results saved to {results_path}")
    return macro_f1
