import torch
from torch.utils.data import Dataset
import numpy as np
from collections import Counter
import logging
from sklearn.feature_extraction.text import TfidfVectorizer # <-- REQUIRED IMPORT

logger = logging.getLogger(__name__)

# ----------------------------------------------------
# START: BASELINE A, D, E Preprocessor (TF-IDF + ML Models)
# ----------------------------------------------------

class TFIDFLogRegPreprocessor:
    """
    TF-IDF Vectorizer wrapper for Logistic Regression, Naive Bayes, and Random Forest models.
    """
    def __init__(self, max_features=50000, ngram_range=(1,2), token_pattern=r"\b\w+\b"):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.token_pattern = token_pattern
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            token_pattern=self.token_pattern
        )

    def fit(self, X):
        """Fit the TF-IDF vectorizer on the training data."""
        self.vectorizer.fit(X)
        return self

    def transform(self, X):
        """Transform text data into TF-IDF feature vectors."""
        return self.vectorizer.transform(X)
    
    def fit_transform(self, X):
        """Fit and then transform the training data."""
        return self.vectorizer.fit_transform(X)

# ----------------------------------------------------
# END: BASELINE A, D, E Preprocessor (TF-IDF + ML Models)
# ----------------------------------------------------


# ----------------------------------------------------
# START: BASELINE B Preprocessor & Utilities (Bi-LSTM)
# ----------------------------------------------------

class CodeDataset(Dataset):
    """Custom PyTorch Dataset for Bi-LSTM input."""
    def __init__(self, texts, labels=None, sequences=None):
        self.texts = texts
        self.labels = labels
        self.sequences = sequences

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {}
        # Sequences are already padded/truncated NumPy arrays from transform()
        if self.sequences is not None:
            item['sequence'] = self.sequences[idx]
        if self.labels is not None:
            item['label'] = self.labels[idx]
        return item

class BiLSTMPreprocessor:
    """
    Preprocessor for Bi-LSTM models.
    Handles tokenization, vocabulary creation, and padding/truncation to max_length.
    """
    def __init__(self, max_length=128, min_freq=5, unk_token='<UNK>', pad_token='<PAD>'):
        self.max_length = max_length
        self.min_freq = min_freq
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.word_to_idx = {self.pad_token: 0, self.unk_token: 1}
        self.idx_to_word = {0: self.pad_token, 1: self.unk_token}
        self.vocab_size = 2

    def _tokenize(self, texts):
        """Simple whitespace tokenization."""
        return [text.split() for text in texts]

    def fit(self, texts):
        """Builds the vocabulary from training texts."""
        tokenized_texts = self._tokenize(texts)
        all_tokens = [token for sublist in tokenized_texts for token in sublist]
        token_counts = Counter(all_tokens)

        # Build vocabulary based on minimum frequency
        for token, count in token_counts.items():
            if count >= self.min_freq:
                if token not in self.word_to_idx:
                    idx = len(self.word_to_idx)
                    self.word_to_idx[token] = idx
                    self.idx_to_word[idx] = token
        
        self.vocab_size = len(self.word_to_idx)
        logger.info(f"Vocabulary built. Size: {self.vocab_size}")

    def transform(self, texts):
        """Converts texts to padded sequences of integer IDs (your preferred logic)."""
        tokenized_texts = self._tokenize(texts)
        sequences = []
        unk_idx = self.word_to_idx[self.unk_token]
        pad_idx = self.word_to_idx[self.pad_token]

        for tokens in tokenized_texts:
            # Convert tokens to indices
            indices = [self.word_to_idx.get(token, unk_idx) for token in tokens]
            
            # 1. Truncate
            indices = indices[:self.max_length]
            
            # 2. Pad
            padding = [pad_idx] * (self.max_length - len(indices))
            padded_indices = indices + padding
            
            sequences.append(padded_indices)

        # Return as NumPy array, as used in train_bilstm_model setup
        return np.array(sequences, dtype=np.int64)

def collate_fn_bilstm(batch):
    """
    FIXED: Optimized Collation function for Bi-LSTM DataLoader. 
    Uses numpy stacking before tensor creation to fix the performance warning.
    """
    # 1. Extract sequences and labels lists
    sequences_list = [item['sequence'] for item in batch]
    labels_list = [item['label'] for item in batch]
    
    # 2. Convert list of NumPy arrays into a single NumPy array (FAST)
    sequences_np = np.array(sequences_list)
    labels_np = np.array(labels_list)
    
    # 3. Convert NumPy arrays to PyTorch tensors (FAST)
    sequences = torch.tensor(sequences_np, dtype=torch.long)
    labels = torch.tensor(labels_np, dtype=torch.long)
    
    return sequences, labels

# ----------------------------------------------------
# END: BASELINE B Preprocessor & Utilities (Bi-LSTM)
# ----------------------------------------------------