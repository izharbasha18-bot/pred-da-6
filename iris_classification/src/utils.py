"""
Utility functions for the Iris Classification project
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, 
                             accuracy_score, precision_score, recall_score, f1_score)

# Configure plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def create_directories():
    """Create necessary directories if they don't exist"""
    dirs = ['models', 'results', 'data']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
    print("✓ Directories created/verified")


def save_model(model, filename='models/trained_model.pkl'):
    """Save trained model to disk"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to {filename}")


def load_model(filename='models/trained_model.pkl'):
    """Load trained model from disk"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model file not found: {filename}")
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"✓ Model loaded from {filename}")
    return model


def save_scaler(scaler, filename='models/scaler.pkl'):
    """Save feature scaler to disk"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved to {filename}")


def load_scaler(filename='models/scaler.pkl'):
    """Load feature scaler from disk"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Scaler file not found: {filename}")
    with open(filename, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✓ Scaler loaded from {filename}")
    return scaler


def plot_confusion_matrix(y_true, y_pred, class_names, filename='results/confusion_matrix.png'):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Iris Classification', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {filename}")
    plt.close()


def plot_feature_importance(model, feature_names, filename='results/feature_importance.png'):
    """Plot and save feature importance"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.ylabel('Importance', fontsize=12)
    plt.xlabel('Features', fontsize=12)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Feature importance plot saved to {filename}")
    plt.close()


def generate_classification_report(y_true, y_pred, class_names, 
                                   filename='results/classification_report.txt'):
    """Generate and save detailed classification report"""
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("IRIS CLASSIFICATION - DETAILED REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(report)
    
    print(f"✓ Classification report saved to {filename}")
    return report


def generate_performance_summary(y_true, y_pred, class_names,
                                filename='results/model_performance.txt'):
    """Generate and save performance summary"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("IRIS CLASSIFICATION - PERFORMANCE METRICS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n")
        f.write(f"\nDataset Size: {len(y_true)} samples\n")
        f.write(f"Test Samples: {len(y_true)}\n")
    
    print(f"✓ Performance summary saved to {filename}")
    
    # Also print to console
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("="*70)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def plot_data_distribution(X, y, feature_names, class_names, 
                          filename='results/data_distribution.png'):
    """Plot distribution of features by class"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Iris Feature Distribution by Species', fontsize=14, fontweight='bold')
    
    axes = axes.ravel()
    for idx, feature_name in enumerate(feature_names):
        for class_idx, class_name in enumerate(class_names):
            mask = y == class_idx
            axes[idx].hist(X[mask, idx], alpha=0.5, label=class_name, bins=15)
        
        axes[idx].set_xlabel(feature_name, fontsize=11)
        axes[idx].set_ylabel('Frequency', fontsize=11)
        axes[idx].set_title(feature_name, fontsize=12, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Data distribution plot saved to {filename}")
    plt.close()


def print_section(title, length=70):
    """Print a formatted section header"""
    print("\n" + "="*length)
    print(title.center(length))
    print("="*length)
