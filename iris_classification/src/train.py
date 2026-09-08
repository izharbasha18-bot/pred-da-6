"""
Training script for Iris Classification model
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from utils import (create_directories, save_model, save_scaler, 
                   plot_data_distribution, print_section)


def load_data():
    """Load and prepare Iris dataset"""
    print_section("LOADING DATA")
    
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    print(f"✓ Dataset loaded successfully")
    print(f"  - Features: {len(feature_names)}")
    print(f"  - Classes: {len(target_names)}")
    print(f"  - Total samples: {len(X)}")
    print(f"\nFeatures:")
    for i, name in enumerate(feature_names):
        print(f"  {i+1}. {name}")
    print(f"\nClasses:")
    for i, name in enumerate(target_names):
        print(f"  {i+1}. {name}")
    
    return X, y, feature_names, target_names


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    print_section("SPLITTING DATA")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✓ Data split completed")
    print(f"  - Training samples: {len(X_train)} ({(len(X_train)/len(X))*100:.1f}%)")
    print(f"  - Testing samples: {len(X_test)} ({(len(X_test)/len(X))*100:.1f}%)")
    print(f"\nClass distribution in training set:")
    for cls_idx in range(len(np.unique(y_train))):
        count = np.sum(y_train == cls_idx)
        print(f"  Class {cls_idx}: {count} samples")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """Scale features using StandardScaler"""
    print_section("SCALING FEATURES")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Features scaled successfully")
    print(f"  - Scaler: StandardScaler")
    print(f"  - Training set statistics:")
    print(f"    Mean: {X_train_scaled.mean(axis=0).round(4)}")
    print(f"    Std:  {X_train_scaled.std(axis=0).round(4)}")
    
    return X_train_scaled, X_test_scaled, scaler


def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """Train Random Forest Classifier"""
    print_section("TRAINING MODEL")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    
    print(f"Training Random Forest Classifier...")
    print(f"  - Number of trees: {n_estimators}")
    print(f"  - Random state: {random_state}")
    
    model.fit(X_train, y_train)
    
    # Get training accuracy
    train_accuracy = model.score(X_train, y_train)
    
    print(f"✓ Model trained successfully")
    print(f"  - Training accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"  - Number of features: {model.n_features_in_}")
    print(f"  - Classes: {model.classes_}")
    
    return model


def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("IRIS CLASSIFICATION - TRAINING PIPELINE".center(70))
    print("="*70)
    
    # Create directories
    create_directories()
    
    # Load data
    X, y, feature_names, target_names = load_data()
    
    # Plot data distribution
    print_section("VISUALIZING DATA")
    plot_data_distribution(X, y, feature_names, target_names)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Train model
    model = train_model(X_train_scaled, y_train)
    
    # Save model and scaler
    print_section("SAVING ARTIFACTS")
    save_model(model)
    save_scaler(scaler)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY".center(70))
    print("="*70)
    print("\nNext steps:")
    print("  1. Run testing script: python test.py")
    print("  2. Check results in 'results/' folder")
    print("  3. View generated visualizations\n")
    
    return model, scaler, X_test_scaled, y_test, feature_names, target_names


if __name__ == "__main__":
    train_model_obj, scaler_obj, X_test, y_test, feat_names, target_names = main()
