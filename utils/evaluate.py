# utils/evaluate.py
import numpy as np
import os
import json
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import seaborn as sns

logger = logging.getLogger(__name__)

def evaluate_and_plot(y_true, y_pred, output_dir, num_labels, model_name='model'):
    """
    Calculates Macro F1 score, saves the classification report, 
    and plots the confusion matrix.

    Args:
        y_true (np.array): True integer labels.
        y_pred (np.array): Predicted integer labels.
        output_dir (str): Directory to save plots and reports.
        num_labels (int): Total number of classes.
        model_name (str): Identifier for the output files.

    Returns:
        float: The Macro F1 score.
    """
    # 1. Calculate Macro F1
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # 2. Generate Classification Report
    report = classification_report(
        y_true, y_pred, 
        labels=range(num_labels), 
        target_names=[str(i) for i in range(num_labels)], 
        output_dict=True, 
        zero_division=0
    )
    report_path = os.path.join(output_dir, f'{model_name}_classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    logger.info(f"Classification report saved to {report_path}")

    # 3. Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(num_labels))
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=range(num_labels), 
        yticklabels=range(num_labels)
    )
    plt.title(f'Confusion Matrix for {model_name.replace("_", " ").title()}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the plot as PDF
    plot_path = os.path.join(output_dir, 'plots', f'{model_name}_confusion_matrix.pdf')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Confusion Matrix saved to {plot_path}")

    return macro_f1