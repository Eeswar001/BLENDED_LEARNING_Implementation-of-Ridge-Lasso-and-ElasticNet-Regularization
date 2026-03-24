# BLENDED_LEARNING
# Implementation of Ridge, Lasso, and ElasticNet Regularization for Predicting Car Price

## AIM:
To implement Ridge, Lasso, and ElasticNet regularization models using polynomial features and pipelines to predict car price.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary libraries.
2. Load the dataset.
3. Preprocess the data (handle missing values, encode categorical variables).
4. Split the data into features (X) and target (y).
5. Divide the data into training and testing sets.
6. Create an SGD Regressor model.
7. Fit the model on the training data.
8. Evaluate the model performance.
9. Make predictions and visualize the results.
## Program:
```
/*
Program to implement SGD Regressor for linear regression.
Developed by: PRAGATHEESWARAN K
RegisterNumber: 212225040310

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Load the dataset
data=pd.read_csv("CarPrice_Assignment.csv")
print(data.head())
print(data.info())

# Data preprocessing
# Dropping unnecessary columns and handling categorical variables
data=data.drop(['CarName','car_ID'],axis=1)
data=pd.get_dummies(data,drop_first=True)

# Splitting the data into features and target variable
x=data.drop('price',axis=1)
y=data['price']

# Standardizing the data
scaler = StandardScaler()
x=scaler.fit_transform(x)
y=scaler.fit_transform(np.array(y).reshape(-1,1))

# Splitting the dataset into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# Creating the SGD Regressor model
sgd_model = SGDRegressor(max_iter=1000,tol=1e-3)

# Fitting the model on the training data
sgd_model.fit(x_train,y_train)

# Making predictions
y_pred=sgd_model.predict(x_test)

# Evaluating model performance
mse=mean_squared_error(y_test,y_pred)

print("="*50)
print('Name: PRAGATHEESWARAN K')
print('Reg No:212225040310')
print(f"MSE: {mse:.4f}")
print(f"R²: {r2_score(y_test,y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred):.4f}")
print("="*50)

# Print model coefficients
print("Model Coefficients:")
print("Coefficiens:",sgd_model.coef_)
print("Intercept:",sgd_model.intercept_)

# Visualizing actual vs predicted prices
plt.scatter(y_test,y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices using SGD Regressor")
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color='red')
plt.show()
*/
```

## Output:
<img width="704" height="503" alt="image" src="https://github.com/user-attachments/assets/21e866a6-03e4-48b9-889d-18b879fdb55e" />
<img width="476" height="583" alt="image" src="https://github.com/user-attachments/assets/ba87ae80-571a-4a4f-89a2-3273447c248e" />
<img width="817" height="289" alt="image" src="https://github.com/user-attachments/assets/23d171a6-c213-477f-8472-82eee1142c39" />
<img width="664" height="454" alt="image" src="https://github.com/user-attachments/assets/a333dd8d-51cc-4d9d-8f67-96866e959083" />

## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
