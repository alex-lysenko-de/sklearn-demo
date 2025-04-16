import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Load dataset
df = pd.read_csv("gender_classification.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "MLP Classifier": make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    )
}

# Store results
results = {}

# Train, predict, and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {
        "accuracy": acc,
        "predictions": y_pred,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=model.classes_ if hasattr(model, "classes_") else model.named_steps['mlpclassifier'].classes_)
    }
    print(f"{name} Accuracy: {acc:.2f}")

# Plot confusion matrices
for name, result in results.items():
    plt.figure(figsize=(6, 4))
    sns.heatmap(result["confusion_matrix"], annot=True, fmt='d', cmap='Blues',
                xticklabels=models[name].classes_ if hasattr(models[name], "classes_") else models[name].named_steps['mlpclassifier'].classes_,
                yticklabels=models[name].classes_ if hasattr(models[name], "classes_") else models[name].named_steps['mlpclassifier'].classes_)
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

# Bar chart comparison
correct_counts = {
    name: (y_test == result["predictions"]).sum()
    for name, result in results.items()
}
incorrect_counts = {
    name: (y_test != result["predictions"]).sum()
    for name, result in results.items()
}

plt.figure(figsize=(8, 5))
bar_width = 0.35
index = range(len(models))

plt.bar(index, list(correct_counts.values()), bar_width, label='Correct', color='green')
plt.bar([i + bar_width for i in index], list(incorrect_counts.values()), bar_width, label='Incorrect', color='red')

plt.xlabel('Model')
plt.ylabel('Count')
plt.title('Correct vs Incorrect Predictions by Model')
plt.xticks([i + bar_width / 2 for i in index], list(models.keys()))
plt.legend()
plt.tight_layout()
plt.show()
