# linear_regression_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class LinearRegressionModel:
    def __init__(self, data, learning_rate):

        self.dataset = data
        self.data = self.dataset.dropna()  # Remove rows with missing values
        self.learning_rate = learning_rate

        # Initialize metrics and plot storage
        self.m = None
        self.b = None
        self.mse = None
        self.mae = None
        self.r2 = None
        self.fig = None

    def gradientDescent(self, x, y, iterations):
        m = 0  # Initial slope
        b = 0  # Initial intercept
        n = x.shape[0]  # Number of data points

        for _ in range(iterations):
            y_pred = m * x + b  # Predicted y values
            dldb = (-2 / n) * sum(y - y_pred)  # Gradient of loss w.r.t. intercept
            dldm = (-2 / n) * sum((y - y_pred) * x)  # Gradient of loss w.r.t. slope

            m -= self.learning_rate * dldm  # Update the slope
            b -= self.learning_rate * dldb  # Update the intercept

            if np.isnan(m) or np.isnan(b):
                print("Gradient descent encountered NaN values. Try reducing the learning rate.")
                return None, None
        
        return m, b

    def trainAndTestModel(self, feature, label):
        x_train, x_test, y_train, y_test = train_test_split(
            self.data[[feature]], self.data[[label]], train_size=0.8, test_size=0.2, random_state=1)

        m, b = self.gradientDescent(x_train.values.flatten(), y_train.values.flatten(), 1000)
        if m is None or b is None:
            return

        self.m, self.b = m, b  # Store the slope and intercept

        y_pred = m * x_test.values.flatten() + b

        # Compute and store error metrics
        self.mse = self.calculateMSE(y_test.values.flatten(), y_pred)
        self.mae = self.calculateMAE(y_test.values.flatten(), y_pred)
        self.r2 = self.calculateR2(y_test.values.flatten(), y_pred)

        # Generate the plot and store it
        self.fig = self.plotRegressionLine(x_test, y_test, y_pred)

    def calculateMSE(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def calculateMAE(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def calculateR2(self, y_true, y_pred):
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)

    def plotRegressionLine(self, x_test, y_test, y_pred):
        fig, ax = plt.subplots()
        ax.scatter(x_test, y_test, color='blue', label='Actual data')
        ax.plot(x_test, y_pred, color='red', label='Fitted line')
        ax.set_xlabel("Feature")
        ax.set_ylabel("Label")
        ax.legend()
        return fig  # Return the figure to be displayed in Streamlit
