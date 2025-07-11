import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import StandardScaler
import joblib

# Load and process the dataset

heart_data = pd.read_csv('data.csv')
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Standardize the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train all models

# Logistic Regression
logistic_model = LogisticRegression(max_iter=1000, random_state=2)
logistic_model.fit(X_train_scaled, Y_train)

# Decision Tree
tree_model = DecisionTreeClassifier(random_state=2)
tree_model.fit(X_train, Y_train)

# Random Forest
forest_model = RandomForestClassifier(random_state=2)
forest_model.fit(X_train, Y_train)

# SVM
svm_model = SVC(kernel="linear", C=3.0, random_state=2)
svm_model.fit(X_train_scaled, Y_train)

# Predictions and accuracy for SVM model
print("SVM Model:")
# Predictions on training data
prediction_on_training_data = svm_model.predict(X_train_scaled)
accuracy_train = accuracy_score(Y_train, prediction_on_training_data)
precision_train = precision_score(Y_train, prediction_on_training_data)
print("Accuracy on training data:", accuracy_train)
print("Precision on training data:", precision_train)

# Predictions on testing data
prediction_on_test_data = svm_model.predict(X_test_scaled)
accuracy_test = accuracy_score(Y_test, prediction_on_test_data)
precision_test = precision_score(Y_test, prediction_on_test_data)
print("Accuracy on testing data:", accuracy_test)
print("Precision on testing data:", precision_test)
print()

# Predictions and accuracy for Logistic Regression model
print("Logistic Regression Model:")
# Predictions on training data
prediction_on_training_data = logistic_model.predict(X_train_scaled)
accuracy_train = accuracy_score(Y_train, prediction_on_training_data)
precision_train = precision_score(Y_train, prediction_on_training_data)
print("Accuracy on training data:", accuracy_train)
print("Precision on training data:", precision_train)

# Predictions on testing data
prediction_on_test_data = logistic_model.predict(X_test_scaled)
accuracy_test = accuracy_score(Y_test, prediction_on_test_data)
precision_test = precision_score(Y_test, prediction_on_test_data)
print("Accuracy on testing data:", accuracy_test)
print("Precision on testing data:", precision_test)
print()

# Predictions and accuracy for Decision Tree model
print("Decision Tree Model:")
# Predictions on training data
prediction_on_training_data = tree_model.predict(X_train)
accuracy_train = accuracy_score(Y_train, prediction_on_training_data)
precision_train = precision_score(Y_train, prediction_on_training_data)
print("Accuracy on training data:", accuracy_train)
print("Precision on training data:", precision_train)

# Predictions on testing data
prediction_on_test_data = tree_model.predict(X_test)
accuracy_test = accuracy_score(Y_test, prediction_on_test_data)
precision_test = precision_score(Y_test, prediction_on_test_data)
print("Accuracy on testing data:", accuracy_test)
print("Precision on testing data:", precision_test)

# Save the SVM model as .pkl file
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("SVM Model saved successfully.")
