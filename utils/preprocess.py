import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
import torch
import pandas as pd
from torch.utils.data import TensorDataset
import numpy as np



# -----------------------------
# Common utilities
# -----------------------------
def load_parquet(path):
    """Reads a parquet file and returns dataframe."""
    return pd.read_parquet(path)

def extract_xy(df):
    """
    df: DataFrame with columns ["code", "label"]
    returns: list(texts), list(labels)
    """
    texts = df["code"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    return texts, labels


# =============================
# TF-IDF preprocessing
# FOR SKLEARN
# =============================

def fit_vectorizer(train_texts, max_features=20000):
    """Fits TF-IDF on training texts only."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        token_pattern=r"(?u)\b\w+\b"   # handles code tokens better
    )
    vectorizer.fit(train_texts)
    return vectorizer

def transform_texts(vectorizer, texts):
    """Transforms texts → TF-IDF → tensor."""
    X = vectorizer.transform(texts).toarray()
    return torch.tensor(X, dtype=torch.float32)


def preprocess_tfidf(train_path, val_path, save_dir="vectorizer"):
    """
    Load train/val parquet → TF-IDF → NumPy arrays.
    Saves vectorizer for future inference.
    """
    train_df = load_parquet(train_path)
    val_df = load_parquet(val_path)

    train_texts, train_labels = extract_xy(train_df)
    val_texts, val_labels = extract_xy(val_df)

    vectorizer = fit_vectorizer(train_texts)

    # Save vectorizer
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(vectorizer, f"{save_dir}/tfidf.pkl")

    X_train = transform_texts(vectorizer, train_texts)
    X_val = transform_texts(vectorizer, val_texts)

    y_train = np.array(train_labels)
    y_val = np.array(val_labels)

    return X_train, y_train, X_val, y_val, vectorizer

# =============================
# TF-IDF preprocessing
# FOR PYTORCHHH
# =============================

def preprocess_tfidf_pyTorch(train_path, val_path, save_dir="vectorizer"):
    """
    Load train/val parquet → TF-IDF → tensors.
    Saves vectorizer for future inference.
    """
    train_df = load_parquet(train_path)
    val_df = load_parquet(val_path)

    train_texts, train_labels = extract_xy(train_df)
    val_texts, val_labels = extract_xy(val_df)

    vectorizer = fit_vectorizer(train_texts)

    # Save vectorizer
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(vectorizer, f"{save_dir}/tfidf.pkl")

    X_train = transform_texts(vectorizer, train_texts)
    X_val = transform_texts(vectorizer, val_texts)

    y_train = torch.tensor(train_labels, dtype=torch.float32)
    y_val = torch.tensor(val_labels, dtype=torch.float32)

    return X_train, y_train, X_val, y_val, vectorizer



# =============================
# Tokenization preprocessing (for TextCNN / LSTM)
# PYTORCH
# =============================
class Tokenizer:
    """Simple word-level tokenizer with vocab building."""
    def __init__(self, min_freq=1):
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.freq = {}
        self.min_freq = min_freq

    def build_vocab(self, texts):
        """Build vocabulary from list of strings"""
        for text in texts:
            for tok in text.split():  # simple whitespace tokenizer
                self.freq[tok] = self.freq.get(tok, 0) + 1
        for tok, count in self.freq.items():
            if count >= self.min_freq and tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
        return self.vocab

    def text_to_ids(self, text):
        """Convert a text string to list of token IDs"""
        return torch.tensor([self.vocab.get(tok, self.vocab["<UNK>"]) for tok in text.split()],
                            dtype=torch.long)

def tokenize_and_pad(texts, tokenizer):
    """Tokenize and pad a list of texts"""
    seqs = [tokenizer.text_to_ids(t) for t in texts]
    padded = pad_sequence(seqs, batch_first=True, padding_value=tokenizer.vocab["<PAD>"])
    return padded

def preprocess_tokenized(train_path, val_path, min_freq=1):
    """
    Reads parquet train + val files, builds vocab on train, tokenizes and pads sequences.
    Returns:
        X_train, y_train, X_val, y_val: tensors
        tokenizer: fitted tokenizer object
    """
    train_texts, train_labels = extract_xy(load_parquet(train_path))
    val_texts, val_labels = extract_xy(load_parquet(val_path))

    tokenizer = Tokenizer(min_freq=min_freq)
    tokenizer.build_vocab(train_texts)

    X_train = tokenize_and_pad(train_texts, tokenizer)
    y_train = torch.tensor(train_labels, dtype=torch.float32)

    X_val = tokenize_and_pad(val_texts, tokenizer)
    y_val = torch.tensor(val_labels, dtype=torch.float32)

    return X_train, y_train, X_val, y_val, tokenizer


# CODE BERT TOKENIZATION 
# FOR PYTORCH

def preprocess_codebert(train_path, val_path, max_length=256):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    def load_data(path):
        df = pd.read_parquet(path)
        texts = df["code"].astype(str).tolist()
        labels = torch.tensor(df["label"].astype(int).tolist(), dtype=torch.float32)
        encodings = tokenizer(texts, truncation=True, padding='max_length',
                              max_length=max_length, return_tensors='pt')
        return encodings, labels

    train_encodings, y_train = load_data(train_path)
    val_encodings, y_val = load_data(val_path)

    # Create TensorDataset
    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], y_train)
    val_dataset   = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], y_val)

    return train_dataset, val_dataset, tokenizer
