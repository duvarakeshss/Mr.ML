import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class Knn_Classification:
    def __init__(self, k, data, features, label):
        self.k = k
        self.data = data
        self.features = features
        self.label = label
    
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, x_test):
        predictions = []
        for test_point in x_test.values:
            distances = [self.euclidean_distance(test_point, x_train_point) for x_train_point in self.x_train.values]
            k_nearest_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train.iloc[i] for i in k_nearest_indices]
            most_common_label = max(k_nearest_labels, key=k_nearest_labels.count)
            predictions.append(most_common_label)
        return predictions

    def evaluate(self):
        x = self.data[self.features]
        y = self.data[self.label]
        
        # Split the data into training and testing sets
        self.x_train, x_test, self.y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        
        # Make predictions on the test set
        predictions = self.predict(x_test)
        
        # Calculate accuracy, confusion matrix, and classification report
        accuracy = accuracy_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        report = classification_report(y_test, predictions)
        
        return accuracy, conf_matrix, report
