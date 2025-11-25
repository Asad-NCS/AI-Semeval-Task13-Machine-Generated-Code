import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import numpy as np
import os
import json
import logging
from tqdm import tqdm
# Import the custom CodeDataset and collate_fn_bilstm from utils/preprocess.py
from utils.preprocess import CodeDataset, collate_fn_bilstm

# Setup logging
logger = logging.getLogger(__name__)

# --- Model Definition ---
class BiLSTMClassifier(nn.Module):
    # INCREASED DROPOUT to 0.6 for better regularization
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout_rate=0.6): 
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        # Multiply hidden_dim by 2 for bidirectional LSTM output
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Take the last layer's forward and backward hidden states
        # hidden is shape: [num_layers * 2, batch_size, hidden_dim]
        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]
        
        # Concatenate them
        combined_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        
        dropped = self.dropout(combined_hidden)
        logits = self.fc(dropped)
        return logits


# --- Training Function ---
def train_bilstm_model(preprocessor, X_train, y_train, X_val, y_val, num_labels, output_dir, 
                       num_epochs=5, batch_size=32, learning_rate=1e-3): # Reverted epochs to 5

    # 1. Device Configuration - GPU CHECK
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {DEVICE}")
    
    # 2. Preprocessing & DataLoaders
    # Transform texts to sequences
    # NOTE: The BiLSTMPreprocessor must be fitted externally before calling this function
    X_train_seq = preprocessor.transform(X_train)
    X_val_seq = preprocessor.transform(X_val)

    # Wrap data in custom PyTorch Dataset (sequences were created in preprocessor.transform)
    train_dataset = CodeDataset(texts=X_train, labels=y_train, sequences=X_train_seq)
    val_dataset = CodeDataset(texts=X_val, labels=y_val, sequences=X_val_seq)
    
    # Use the custom collate_fn_bilstm for padding/stacking
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_bilstm)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_bilstm)

    # 3. Model Setup
    # FIXED: Access the vocab size correctly from the BiLSTMPreprocessor attribute
    MODEL_VOCAB_SIZE = preprocessor.vocab_size 
    
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    
    model = BiLSTMClassifier(
        vocab_size=MODEL_VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=num_labels
    ).to(DEVICE) # MOVE MODEL TO DEVICE

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    logger.info("Starting Bi-LSTM training...")
    
    best_f1 = 0.0
    
    # 4. Training Loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        
        # Use tqdm for a visible progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs} [Train]', unit='batch')
        for sequences, labels in pbar:
            # MOVE DATA TO DEVICE
            sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        # 5. Evaluation
        model.eval()
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                # MOVE DATA TO DEVICE
                sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(sequences)
                _, predicted = torch.max(outputs.data, 1)
                
                # Move predictions/labels back to CPU for metric calculation
                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        # Calculate metrics
        macro_f1 = f1_score(val_true, val_preds, average='macro')
        
        logger.info(f"Epoch {epoch}/{num_epochs}: Validation Macro F1: {macro_f1:.4f}")

        # Save the best model
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            # Note: Ensure the directory exists before saving
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            torch.save(model.state_dict(), os.path.join(output_dir, 'bilstm_best_model.pth'))
            logger.info(f"New best model saved with F1: {best_f1:.4f}")

    # 6. Final Evaluation and Report
    logger.info("--- Bi-LSTM Training Complete ---")
    
    # Reload the best model for final evaluation
    model_path = os.path.join(output_dir, 'bilstm_best_model.pth')
    if os.path.exists(model_path):
        # We need to explicitly load the state dict to the correct device (DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    
    model.eval()
    
    # Recalculate final metrics on validation set using the best model
    val_preds = []
    val_true = []
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 1)
            val_preds.extend(predicted.cpu().numpy())
            val_true.extend(labels.cpu().numpy())
            
    final_f1 = f1_score(val_true, val_preds, average='macro')
    report = classification_report(val_true, val_preds, output_dict=True)
    conf_matrix = confusion_matrix(val_true, val_preds)

    # Save results
    with open(os.path.join(output_dir, 'baselineB_classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
        
    np.savetxt(os.path.join(output_dir, 'baselineB_confusion_matrix.txt'), conf_matrix, fmt='%d')

    logger.info(f"[Baseline B] Final Macro F1 on validation set: {final_f1:.4f}")
    
    return final_f1