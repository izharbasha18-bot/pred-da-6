"""
Testing script for Iris Classification model
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import (load_model, load_scaler, plot_confusion_matrix,
                   plot_feature_importance, generate_classification_report,
                   generate_performance_summary, plot_data_distribution, print_section)


def load_test_data(test_size=0.2, random_state=42):
    """Load and prepare test data"""
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Split to get same test set as training
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_test, y_test, feature_names, target_names


def main():
    """Main testing pipeline"""
    print("\n" + "="*70)
    print("IRIS CLASSIFICATION - TESTING PIPELINE".center(70))
    print("="*70)
    
    # Load model and scaler
    print_section("LOADING ARTIFACTS")
    try:
        model = load_model()
        scaler = load_scaler()
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease run training first:")
        print("  python train.py")
        return
    
    # Load test data
    print_section("LOADING TEST DATA")
    X_test, y_test, feature_names, target_names = load_test_data()
    print(f"✓ Test data loaded: {len(X_test)} samples")
    print(f"  Features: {list(feature_names)}")
    print(f"  Classes: {list(target_names)}")
    
    # Scale test data
    print_section("SCALING TEST DATA")
    X_test_scaled = scaler.transform(X_test)
    print(f"✓ Test data scaled successfully")
    
    # Make predictions
    print_section("MAKING PREDICTIONS")
    y_pred = model.predict(X_test_scaled)
    print(f"✓ Predictions completed for {len(y_pred)} test samples")
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  - Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Generate visualizations and reports
    print_section("GENERATING RESULTS")
    
    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, target_names)
    
    # Feature importance
    plot_feature_importance(model, feature_names)
    
    # Classification report
    report = generate_classification_report(y_test, y_pred, target_names)
    print("\n" + report)
    
    # Performance summary
    metrics = generate_performance_summary(y_test, y_pred, target_names)
    
    # Detailed predictions
    print_section("SAMPLE PREDICTIONS")
    print(f"\nShowing first 10 test samples:\n")
    print(f"{'Sample':<8} {'Sepal L':<10} {'Sepal W':<10} {'Petal L':<10} {'Petal W':<10} {'True':<15} {'Pred':<15} {'Correct':<10}")
    print("-" * 110)
    
    for i in range(min(10, len(X_test))):
        true_label = target_names[y_test[i]]
        pred_label = target_names[y_pred[i]]
        is_correct = "✓" if y_test[i] == y_pred[i] else "✗"
        
        print(f"{i+1:<8} {X_test[i, 0]:<10.2f} {X_test[i, 1]:<10.2f} {X_test[i, 2]:<10.2f} {X_test[i, 3]:<10.2f} {true_label:<15} {pred_label:<15} {is_correct:<10}")
    
    # Error analysis
    print_section("ERROR ANALYSIS")
    errors = y_test != y_pred
    error_count = np.sum(errors)
    
    print(f"Total misclassifications: {error_count}/{len(y_test)}")
    print(f"Error rate: {(error_count/len(y_test))*100:.2f}%")
    
    if error_count > 0:
        print(f"\nMisclassified samples:")
        error_indices = np.where(errors)[0]
        for idx in error_indices[:5]:  # Show first 5 errors
            true_label = target_names[y_test[idx]]
            pred_label = target_names[y_pred[idx]]
            print(f"  Sample {idx}: True={true_label}, Predicted={pred_label}")
    
    # Per-class accuracy
    print_section("PER-CLASS ACCURACY")
    for cls_idx, class_name in enumerate(target_names):
        mask = y_test == cls_idx
        class_accuracy = accuracy_score(y_test[mask], y_pred[mask])
        class_count = np.sum(mask)
        print(f"{class_name:<15} {class_accuracy:.4f} ({class_accuracy*100:.2f}%) - {class_count} samples")
    
    print("\n" + "="*70)
    print("TESTING COMPLETED SUCCESSFULLY".center(70))
    print("="*70)
    print("\nResults saved in 'results/' folder:")
    print("  - confusion_matrix.png")
    print("  - feature_importance.png")
    print("  - classification_report.txt")
    print("  - model_performance.txt\n")


if __name__ == "__main__":
    main()
