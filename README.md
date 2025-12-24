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


## Output:
![logistic regression using gradient descent](sam.png)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

