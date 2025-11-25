# train.py (Combined Orchestrator for Baselines A, B, D, E)
import os
import argparse
import logging
import numpy as np

# Import utility functions
# NOTE: load_baselineA_datasets is used by all TF-IDF models (A, D, E)
from utils.data_loader import load_baselineA_datasets, load_data_and_split
from utils.preprocess import BiLSTMPreprocessor, TFIDFLogRegPreprocessor

# -----------------------------------------------------------------
# START: BASELINE IMPORTS 
# -----------------------------------------------------------------
# Baseline A: TF-IDF + Logistic Regression
from baseline.model_baseline_A import train_tfidf_logreg_model 
# Baseline B: Bi-LSTM
from baseline.model_baseline_B import train_bilstm_model 
# Baseline D: TF-IDF + Naive Bayes (NEW)
from baseline.model_baseline_NB import train_nb_model
# Baseline E: TF-IDF + Random Forest (NEW)
from baseline.model_baseline_RF import train_rf_model
# -----------------------------------------------------------------
# END: BASELINE IMPORTS
# -----------------------------------------------------------------


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration: TASK A Files ---
# All models will use the full training and validation sets as inputs
LABEL_TO_ID_PATH = './data/label_to_id.json'
ID_TO_LABEL_PATH = './data/id_to_label.json'
TRAIN_PARQUET_PATH = './data/task_a_training_set_1.parquet'
VALID_PARQUET_PATH = './data/task_a_validation_set.parquet'
# ------------------------------------

class BaselineRunner:
    def __init__(self, task_subset: str = 'A'):
        self.task_subset = task_subset
        
        # 1. Load Data for TF-IDF Models (A, D, E)
        # Loads X_train/X_val from separate parquet files (needed for clean TF-IDF fitting)
        self.X_train_ML, self.y_train_ML, self.X_val_ML, self.y_val_ML = self.load_ml_data()
        
        # 2. Load Data for Bi-LSTM Model (B)
        # Loads X_train/X_val from the single training file (splits 80/20 internally)
        self.X_train_B, self.X_val_B, self.y_train_B, self.y_val_B, self.num_labels_B = self.load_bilstm_data()


    def load_ml_data(self):
        """Loads data for TF-IDF based models (A, D, E)."""
        logger.info("Loading ML (TF-IDF) data from separate Train/Val files...")
        X_train, y_train, X_val, y_val = load_baselineA_datasets(
            train_parquet_path=TRAIN_PARQUET_PATH,
            valid_parquet_path=VALID_PARQUET_PATH
        )
        if X_train is None:
            raise RuntimeError("Failed to load ML datasets. Check file paths.")
        return X_train, y_train, X_val, y_val

    def load_bilstm_data(self):
        """Loads and splits data for Baseline B (Bi-LSTM)."""
        logger.info("Loading Bi-LSTM data (single file, internal 80/20 split)...")
        # Bi-LSTM loader splits the single training file into train/val internally.
        X_train, X_val, y_train, y_val, num_labels = load_data_and_split(
            parquet_path=TRAIN_PARQUET_PATH, # Using the large training file as input
            label_to_id_path=LABEL_TO_ID_PATH,
            id_to_label_path=ID_TO_LABEL_PATH,
            test_size=0.2,          
            random_state=42
        )
        if X_train is None:
            raise RuntimeError("Failed to load Bi-LSTM data.")
        return X_train, X_val, y_train, y_val, num_labels

    # --- Baseline A: LogReg ---
    def run_tfidf_logreg_baseline(self, output_dir: str):
        """Runs Baseline A: TF-IDF + Logistic Regression"""
        logger.info("\n--- Running Baseline A: TF-IDF + Logistic Regression ---")
        macro_f1 = train_tfidf_logreg_model(self.X_train_ML, self.y_train_ML, self.X_val_ML, self.y_val_ML, output_dir)
        logger.info(f"[Baseline A] Final Macro F1: {macro_f1:.4f}")
        return macro_f1
    
    # --- Baseline D: Naive Bayes ---
    def run_nb_baseline(self, output_dir: str):
        """Runs Baseline D: TF-IDF + Naive Bayes"""
        logger.info("\n--- Running Baseline D: TF-IDF + Naive Bayes ---")
        macro_f1 = train_nb_model(self.X_train_ML, self.y_train_ML, self.X_val_ML, self.y_val_ML, output_dir)
        logger.info(f"[Baseline D] Final Macro F1: {macro_f1:.4f}")
        return macro_f1

    # --- Baseline E: Random Forest ---
    def run_rf_baseline(self, output_dir: str):
        """Runs Baseline E: TF-IDF + Random Forest"""
        logger.info("\n--- Running Baseline E: TF-IDF + Random Forest ---")
        macro_f1 = train_rf_model(self.X_train_ML, self.y_train_ML, self.X_val_ML, self.y_val_ML, output_dir)
        logger.info(f"[Baseline E] Final Macro F1: {macro_f1:.4f}")
        return macro_f1


    # --- Baseline B: Bi-LSTM ---
    def run_bilstm_baseline(self, output_dir: str, **kwargs):
        """Runs Baseline B: Bi-LSTM"""
        logger.info("\n--- Running Baseline B: Bi-LSTM Model ---")
        
        preprocessor = BiLSTMPreprocessor()
        
        final_f1 = train_bilstm_model(
            preprocessor=preprocessor,
            X_train=self.X_train_B,
            y_train=self.y_train_B,
            X_val=self.X_val_B,
            y_val=self.y_val_B,
            num_labels=self.num_labels_B,
            output_dir=output_dir,
            **kwargs 
        )
        logger.info(f"[Baseline B] Final Macro F1: {final_f1:.4f}")
        return final_f1


def main():
    parser = argparse.ArgumentParser(description='Run Baselines for SemEval-2026-Task13')
    parser.add_argument('--task', choices=['A'], default='A', help='Task subset to use (currently only A supported)')
    parser.add_argument('--output_dir', default='./results', help='Output directory for weights and metrics')
    
    # Add a flag to select which model to run
    parser.add_argument('--model', choices=['A', 'B', 'D', 'E', 'all'], default='B', 
                        help='Specify which model to run (A: LogReg, B: BiLSTM, D: NB, E: RF, all: Run all models). Default is B.')
    
    # Bi-LSTM specific arguments
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs for Bi-LSTM')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for Bi-LSTM')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for Bi-LSTM')
    
    args = parser.parse_args()
    
    # Create necessary output folders
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)

    try:
        runner = BaselineRunner(task_subset=args.task)
        
        model_selection = args.model.lower()
        
        if model_selection == 'a' or model_selection == 'all':
            runner.run_tfidf_logreg_baseline(args.output_dir)

        if model_selection == 'd' or model_selection == 'all':
            runner.run_nb_baseline(args.output_dir)

        if model_selection == 'e' or model_selection == 'all':
            runner.run_rf_baseline(args.output_dir)

        if model_selection == 'b' or model_selection == 'all':
            bilstm_params = {
                'num_epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate
            }
            runner.run_bilstm_baseline(args.output_dir, **bilstm_params)

        logger.info("\n=== Assignment 2 Baseline Execution Complete! ===")
        
    except RuntimeError as e:
        logger.error(f"Critical error: {e}")
        exit(1)

if __name__ == '__main__':
    main()

#to activate the virtual environment use
#.\venv_311\Scripts\Activate.ps1