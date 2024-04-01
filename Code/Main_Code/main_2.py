import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the training and test data
train_data = pd.read_csv("C:/Users/parth/PycharmProjects/Guvi_Mentor/Project_3/Dataset/p1_train.csv", header=None, delimiter=',')
test_data = pd.read_csv("C:/Users/parth/PycharmProjects/Guvi_Mentor/Project_3/Dataset/p1_test.csv", header=None, delimiter=',')

# Extract features and target variable from the datasets
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Train the linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict with linear regression model
y_pred_linear = linear_model.predict(X_test)

# Train the Support Vector Regression model
svr_model = SVR(kernel='linear')
svr_model.fit(X_train, y_train)

# Predict with SVR model
y_pred_svr = svr_model.predict(X_test)

# Calculate Mean Squared Error (MSE) and Mean Absolute Error (MAE) for Linear Regression
mse_linear = mean_squared_error(y_test, y_pred_linear)
mae_linear = mean_absolute_error(y_test, y_pred_linear)

print("Linear Regression:")
print("Mean Squared Error (MSE):", mse_linear)
print("Mean Absolute Error (MAE):", mae_linear)

# Calculate Mean Squared Error (MSE) and Mean Absolute Error (MAE) for SVR
mse_svr = mean_squared_error(y_test, y_pred_svr)
mae_svr = mean_absolute_error(y_test, y_pred_svr)

print("\nSupport Vector Regression (SVR):")
print("Mean Squared Error (MSE):", mse_svr)
print("Mean Absolute Error (MAE):", mae_svr)
