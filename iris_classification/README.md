# Iris Flower Classification Project

## 📋 Project Overview
This is a complete machine learning classification project using the famous Iris dataset. The project demonstrates a full ML workflow including data loading, model training, evaluation, and visualization.

## 🎯 Objective
Classify iris flowers into three species (Setosa, Versicolor, Virginica) based on their physical measurements:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

## 📁 Project Structure
```
iris_classification/
├── data/                    # Dataset storage
│   └── iris_data.csv       # Iris dataset
├── models/                  # Trained model storage
│   ├── trained_model.pkl   # Serialized trained model
│   └── scaler.pkl          # Feature scaler
├── results/                 # Output results and visualizations
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── model_performance.txt
├── src/                     # Source code
│   ├── train.py            # Training script
│   ├── test.py             # Testing script
│   └── utils.py            # Utility functions
├── requirements.txt         # Project dependencies
├── README.md               # This file
└── main.py                 # Main execution script
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation
1. Clone or download the project
2. Navigate to the project directory:
   ```bash
   cd iris_classification
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Usage

### Train the Model
```bash
python main.py --mode train
```
This will:
- Load the Iris dataset
- Split data into training (80%) and testing (20%)
- Scale features using StandardScaler
- Train a Random Forest Classifier
- Save the model and scaler

### Test the Model
```bash
python main.py --mode test
```
This will:
- Load the trained model
- Evaluate on test data
- Generate classification metrics
- Create visualizations

### Run Complete Pipeline
```bash
python main.py --mode all
```
This will train and test in one go.

## 📈 Model Performance Metrics
The project outputs:
- **Accuracy**: Overall correctness of predictions
- **Precision**: True positive rate among predicted positives
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visualization of prediction accuracy per class

## 📊 Visualizations Generated
1. **Confusion Matrix**: Shows prediction accuracy per iris species
2. **Feature Importance**: Displays which features are most predictive
3. **Distribution Plots**: Shows data distribution by species
4. **Classification Report**: Detailed metrics in text and visual format

## 🔧 Technical Details

### Model Architecture
- **Algorithm**: Random Forest Classifier
- **n_estimators**: 100 trees
- **test_size**: 0.2 (20% test data)
- **random_state**: 42 (reproducibility)

### Data Preprocessing
- Feature scaling using StandardScaler
- 80-20 train-test split
- No missing values handling (Iris dataset is clean)

## 📁 Output Files
All results are saved in the `results/` folder:
- `confusion_matrix.png`: Visualization of model predictions
- `classification_report.txt`: Detailed performance metrics
- `model_performance.txt`: Summary statistics
- `feature_importance.png`: Feature importance chart

## 🤝 Contributing
Feel free to modify and improve the project:
- Try different ML algorithms
- Experiment with hyperparameters
- Add cross-validation
- Implement custom preprocessing

## 📝 License
This project is free to use for educational purposes.

## ✅ Checklist
- [x] Data loading and exploration
- [x] Train-test data splitting
- [x] Feature scaling
- [x] Model training
- [x] Model evaluation
- [x] Performance visualization
- [x] Results documentation
- [x] Clean code structure

## 🐛 Troubleshooting
**Model not found error**: Run `python main.py --mode train` first
**Missing dependencies**: Run `pip install -r requirements.txt`
**Permission denied**: Ensure you have write access to the project folder

---
**Last Updated**: 2026-09-08
**Author**: AI Assistant
**Python Version**: 3.8+
