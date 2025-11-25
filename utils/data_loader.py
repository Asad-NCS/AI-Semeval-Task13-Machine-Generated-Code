import pandas as pd
import json
from sklearn.model_selection import train_test_split
import logging
import numpy as np

logger = logging.getLogger(__name__)

def _load_single_parquet_data(parquet_path: str):
    """Internal helper to load data from a single parquet file."""
    df = pd.read_parquet(parquet_path)
    # Assuming pre-numericalized labels ('0', '1') are converted to integers
    try:
        df['label_id'] = df['label'].astype(str).astype(int)
    except ValueError:
        logger.error(f"Labels in {parquet_path} are not simple integers (0/1). Check data format.")
        return None, None
        
    X = df['code'].values 
    y = df['label_id'].values 
    return X, y

def load_baselineA_datasets(train_parquet_path: str, valid_parquet_path: str):
    """
    Loads data for Baselines A, D, E (TF-IDF models) where training and
    validation sets are provided in separate parquet files.
    """
    logger.info(f"Loading Training data from {train_parquet_path}...")
    X_train, y_train = _load_single_parquet_data(train_parquet_path)
    
    logger.info(f"Loading Validation data from {valid_parquet_path}...")
    X_val, y_val = _load_single_parquet_data(valid_parquet_path)
    
    if X_train is None or X_val is None:
        logger.error("Failed to load one or both baseline A datasets.")
        return None, None, None, None
        
    logger.info(f"Baseline A/D/E Data Loaded. Train size: {len(X_train)}, Validation size: {len(X_val)}")
    
    return X_train, y_train, X_val, y_val


def load_data_and_split(parquet_path: str, label_to_id_path: str, id_to_label_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Loads data from a single parquet file, handles pre-numericalized labels, 
    and splits data into train/validation sets (used by Bi-LSTM).
    """
    try:
        # 1. Load Data
        df = pd.read_parquet(parquet_path)
        
        # 2. Load Label Maps
        with open(label_to_id_path, 'r') as f:
            # We load this to get num_labels, even if we don't use it for mapping
            label_to_id = json.load(f)
        
        with open(id_to_label_path, 'r') as f:
            id_to_label = json.load(f) 
            
        num_labels = len(label_to_id)
        logger.info(f"Loaded label maps: Found {num_labels} classes.")
        
        # 3. Map labels to integer IDs (FIXED LOGIC)
        try:
            df['label_id'] = df['label'].astype(str).astype(int)
            logger.info("Successfully converted pre-numericalized labels ('0', '1') to integers.")
        except ValueError:
            logger.error("Data labels are not simple integers (0/1). Check data format.")
            return None, None, None, None, None
            
        # 4. Final Label Validation
        valid_labels = set(range(num_labels))
        if not set(df['label_id'].unique()).issubset(valid_labels):
            logger.error("Error: Data labels, after conversion, are outside the expected range (0 to num_labels-1).")
            return None, None, None, None, None

        # 5. Data Split
        X = df['code'].values 
        y = df['label_id'].values 
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        logger.info(f"Data split complete. Train size: {len(X_train)}, Validation size: {len(X_val)}")
        return X_train, X_val, y_train, y_val, num_labels
    
    except Exception as e:
        logger.error(f"Error loading and splitting data: {e}")
        return None, None, None, None, None