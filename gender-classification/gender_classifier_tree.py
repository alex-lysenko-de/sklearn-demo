
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree

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

# 4. Create DecisionTree object
clf = DecisionTreeClassifier()

# 5. Fit the DecisionTree to the training data
clf.fit(X_train, y_train)

# 5b. Visualize the decision tree
plt.figure(figsize=(32, 20), dpi=300)
plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True, rounded=True)
plt.title("Decision Tree Visualization")
plt.show()


# 6. Make a prediction on the test data
y_pred = clf.predict(X_test)
print("Predictions: ", y_pred)  

# 7. Output the accuracy
# Accuracy = (number of correct predictions) / (total number of predictions)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 8. Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 9. Bar Chart of Correct vs Incorrect
correct = (y_test == y_pred).sum()
incorrect = (y_test != y_pred).sum()

plt.figure()
plt.bar(['Correct', 'Incorrect'], [correct, incorrect], color=['green', 'red'])
plt.title('Prediction Results')
plt.ylabel('Count')
plt.show()

