# Gender Classification with Scikit-Learn

This is a simple Python script that demonstrates how to use the `scikit-learn` library to classify gender based on a dataset using a Decision Tree Classifier. It also includes visualizations: a decision tree, a confusion matrix, and a bar chart showing prediction performance.

## 📁 Files

- `gender_classification.csv` – The dataset file
- `compare_models.py` – Python script containing the implementation

## 🛠️ Requirements

Make sure you have the following Python packages installed:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

## 🚀 How to Run

1. Clone or download the repository.
2. Place `gender_classification.csv` in the same directory as `compare_models.py`.
3. Run the script:

```bash
python compare_models.py
```

## 🧠 What the Script Does

1. Loads the dataset as a pandas DataFrame.
2. Splits the data into features and labels.
3. Splits the data into training and test sets.
4. Trains a Decision Tree classifier.
5. Predicts the labels for the test data.
6. Outputs the classification accuracy.
7. Visualizes the decision tree.
8. Displays a confusion matrix.
9. Shows a bar chart with correct vs. incorrect predictions.


## 📚 Libraries Used

- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)


## Screenshot
![decision_tree](img/decision_tree.png)

![confusion_matrix](img/confusion_matrix.png)

![prediction_results](img/prediction_results.png)

