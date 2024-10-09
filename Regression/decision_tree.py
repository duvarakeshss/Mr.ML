# decision_tree_regressor.py
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class DecisionTreeModel:
    def __init__(self):
        self.model = DecisionTreeRegressor(random_state=42)
        self.label_encoders = {}

    def encode_labels(self, data):
        for column in data.columns:
            if data[column].dtype == 'object':
                le = LabelEncoder()
                data[column] = le.fit_transform(data[column].astype(str))
                self.label_encoders[column] = le
        return data

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        return y_test, y_pred

    def evaluate(self, y_test, y_pred):
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mae, mse, r2

    def get_tree(self):
        return self.model
