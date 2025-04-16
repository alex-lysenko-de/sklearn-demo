
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns


# 1. Load the dataset
df = pd.read_csv("gender_classification.csv")

# 2. Split the dataset into features and labels    
# X - is the matrix of features, y - is the vector of labels 
X = df.drop(columns=['gender']) # x == all columns except the last one
y = df['gender'] # y == the last column

# 3. Split into training and test data
# 80% training data, 20% test data  
# random_state is used to reproduce the same results every time (see random.seed())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create MLP object
clf = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=42)
)

# 5. Fit the model to the training data
clf.fit(X_train, y_train)


# 6. Make a prediction on the test data
y_pred = clf.predict(X_test)
print("Predictions: ", y_pred)  

# 7. Output the accuracy
# Accuracy = (number of correct predictions) / (total number of predictions)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Show training loss curve (only works if clf is pipeline with MLP inside)
mlp = clf.named_steps['mlpclassifier']

plt.figure()
plt.plot(mlp.loss_curve_)
plt.title('MLP Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: Visualize weights of the first hidden layer
weights = mlp.coefs_[0]  # Shape: (n_features, n_hidden_neurons)
plt.figure(figsize=(10, 6))
sns.heatmap(weights, cmap="coolwarm", center=0,
            xticklabels=[f"H{i}" for i in range(weights.shape[1])],
            yticklabels=X.columns)
plt.title("Weights from Input to Hidden Layer")
plt.xlabel("Hidden Neurons")
plt.ylabel("Input Features")
plt.tight_layout()
plt.show()
