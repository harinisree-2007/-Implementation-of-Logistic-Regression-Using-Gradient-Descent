# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Data Collection
- Gather dataset with input features X and binary target variable y\in \{ 0,1\} .
Step 2: Data Preprocessing
- Handle missing values.
- Normalize/scale features for faster convergence.
- Encode categorical variables if present.
Step 3: Initialize Parameters
- Initialize weights w and bias b to small random values or zeros.
Step 4: Define Logistic Function
- Logistic (sigmoid) function:
\sigma (z)=\frac{1}{1+e^{-z}}
where z=w^TX+b.
Step 5: Define Cost Function
- Binary cross-entropy loss:

where \hat {y}^{(i)}=\sigma (z^{(i)}).
Step 6: Gradient Descent Updates
- Compute gradients:
- Compute gradients:
\frac{\partial J}{\partial w}=\frac{1}{m}X^T(\hat {y}-y)
\frac{\partial J}{\partial b}=\frac{1}{m}\sum _{i=1}^m(\hat {y}^{(i)}-y^{(i)})
- Update parameters:
w:=w-\alpha \cdot \frac{\partial J}{\partial w}
b:=b-\alpha \cdot \frac{\partial J}{\partial b}
where \alpha  is the learning rate.
Step 7: Iteration
- Repeat gradient descent updates until convergence or max iterations.
Step 8: Prediction
- For new input X, compute \hat {y}=\sigma (w^TX+b).
- Classify as spam (1) if \hat {y}\geq 0.5, else ham (0).
Step 9: Evaluation
- Measure accuracy, precision, recall, F1-score.




## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.

```
import pandas as pd
import numpy as np

# Load and preprocess the data
data = pd.read_csv("Placement_Data.csv")
data1 = data.drop(['sl_no', 'salary'], axis=1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])

# Split features and target
X = data1.iloc[:, :-1].values  # Features
Y = data1["status"].values  # Target variable

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize parameters
theta = np.random.randn(X.shape[1])  # Random initialization
alpha = 0.01  # Learning rate
num_iterations = 1000  # Number of iterations

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define loss function
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15)) / len(y)

# Gradient Descent function
def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta

# Train the model
theta = gradient_descent(theta, X, Y, alpha, num_iterations)

# Prediction function
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    return np.where(h >= 0.5, 1, 0)

# Model evaluation
y_pred = predict(theta, X)
accuracy = np.mean(y_pred == Y)

# Display Results
print("Accuracy:", accuracy)
print("\nPredicted:\n", y_pred)
print("\nActual:\n", Y)

# Predictions for new data
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])  # Example input
xnew = scaler.transform(xnew)  # Apply same scaling as training data
y_prednew = predict(theta, xnew)
print("\nPredicted Result:", y_prednew)

Output:
![logistic regression using gradient descent](sam.png)
<img width="982" height="595" alt="image" src="https://github.com/user-attachments/assets/8913d67e-48a9-4912-90e3-021ce37ff4a0" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

