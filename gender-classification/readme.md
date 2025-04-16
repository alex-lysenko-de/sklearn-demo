---

## 📁 1. `README.md` (Main)

```markdown
# Gender Classification with Scikit-Learn

This project demonstrates how to classify gender using different machine learning models with the `scikit-learn` library in Python.

## 📂 Available Models

1. [🌳 Decision Tree Classifier](./README_decision_tree.md)
2. [🧠 MLP Classifier (Neural Network)](./README_mlp.md)

## 📈 Features

- Loads and prepares data from `gender_classification.csv`
- Compares two classification models
- Displays accuracy, confusion matrix, and prediction results
- Visualizes decision tree structure (for DecisionTreeClassifier)

## 🔁 Bonus

Also included is a script to compare **both models side by side**:
```bash
python compare_models.py
```

It shows:
- Accuracy of both models
- Confusion matrices
- Side-by-side bar chart of prediction performance

---
```

---

## 🌳 2. `README_decision_tree.md`

```markdown
# Decision Tree Classifier 🌳

This script uses a `DecisionTreeClassifier` from `scikit-learn` to classify gender from the dataset.

## 🛠️ Requirements

```bash
pip install pandas scikit-learn matplotlib seaborn
```

## 🚀 How to Run

```bash
python gender_classifier_tree.py
```

## 🧠 What It Does

- Loads data from `gender_classification.csv`
- Splits into train and test sets
- Trains a Decision Tree model
- Prints accuracy
- Shows:
  - Confusion matrix
  - Bar chart of correct vs. incorrect predictions
  - Full decision tree visualization

## 📈 Example Output

```
Accuracy: 0.92
```

## 📊 Visuals

- Confusion Matrix (Heatmap)
- Bar Chart of Correct vs. Incorrect
- Full Tree Diagram using `plot_tree()`

---
```

---

## 🧠 3. `README_mlp.md`

```markdown
# MLP Classifier (Neural Network) 🧠

This script uses `MLPClassifier` (Multi-layer Perceptron) with a `StandardScaler` pipeline to classify gender from the dataset.

## 🛠️ Requirements

```bash
pip install pandas scikit-learn matplotlib seaborn
```

## 🚀 How to Run

```bash
python gender_classifier_mlp.py
```

## 🧠 What It Does

- Loads data from `gender_classification.csv`
- Scales features using `StandardScaler`
- Trains a neural network model
- Prints accuracy
- Shows:
  - Confusion matrix
  - Bar chart of correct vs. incorrect predictions

## ⚙️ Model Details

- One hidden layer with 100 neurons
- Activation: ReLU
- Optimizer: Adam
- Epochs: up to 500

## 📈 Example Output

```
Accuracy: 0.94
```
