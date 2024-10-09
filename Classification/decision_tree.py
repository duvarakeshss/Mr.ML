# decision_tree_classifier.py
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DecisionTreeModel:
    def __init__(self):
        self.model = DecisionTreeClassifier(random_state=42)
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
        return X_test, y_test

    def predict(self, X):
        return self.model.predict(X)

    def get_tree(self):
        return self.model
