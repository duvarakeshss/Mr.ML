# multiple_linear_regression.py
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class MultipleRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        # Add intercept term to the features
        X = np.column_stack((np.ones(X.shape[0]), X))
        X_transpose = X.T
        self.coefficients = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)
        self.intercept = self.coefficients[0]

    def predict(self, X):
        # Add intercept term to the features
        X = np.column_stack((np.ones(X.shape[0]), X))
        return X.dot(self.coefficients)

    def evaluate(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return mae, mse, r2
