import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ReduceLROnPlateau
import pickle

class TuberculosisClassifier:
    def __init__(self, normal_dir, tb_dir, image_size=256):
        self.normal_dir = normal_dir
        self.tb_dir = tb_dir
        self.image_size = image_size
        self.model = None

    def load_and_preprocess_data(self):
        images = []
        labels = []

        # Load normal images
        for x in os.listdir(self.normal_dir):
            imagedir = os.path.join(self.normal_dir, x)
            image = cv.imread(imagedir, cv.IMREAD_GRAYSCALE)
            image = cv.resize(image, (self.image_size, self.image_size))
            images.append(image)
            labels.append(0)

        # Load TB images
        for y in os.listdir(self.tb_dir):
            imagedir = os.path.join(self.tb_dir, y)
            image = cv.imread(imagedir, cv.IMREAD_GRAYSCALE)
            image = cv.resize(image, (self.image_size, self.image_size))
            images.append(image)
            labels.append(1)

        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)

        # Split data into train and test sets
        imagetrain, imagetest, labeltrain, labeltest = train_test_split(images, labels, test_size=0.3, random_state=42)
        imagetrain = (imagetrain.astype('float32')) / 255
        imagetest = (imagetest.astype('float32')) / 255

        # Reshape imagetrain for SMOTE
        imagetrain = imagetrain.reshape(len(imagetrain), (self.image_size * self.image_size))

        # Perform SMOTE
        smote = SMOTE(random_state=42)
        imagetrain, labeltrain = smote.fit_resample(imagetrain, labeltrain)

        # Reshape imagetrain for CNN input
        imagetrain = imagetrain.reshape(-1, self.image_size, self.image_size, 1)

        self.imagetrain = imagetrain
        self.imagetest = imagetest
        self.labeltrain = labeltrain
        self.labeltest = labeltest

    def build_model(self):
        # Define the CNN model
        self.model = tf.keras.Sequential([
            layers.InputLayer(input_shape=(self.image_size, self.image_size, 1)),
            Conv2D(16, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        # Compile the model
        self.model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy']
        )

    def train_model(self, batch_size=16, epochs=10):
        # Callback for learning rate reduction
        reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=0.1, patience=1, min_lr=0.00001, verbose=1)

        # Train the model
        self.model.fit(self.imagetrain, self.labeltrain, batch_size=batch_size, epochs=epochs, verbose=2, callbacks=[reduce_lr])

    def evaluate_model(self):
        # Evaluate the model
        print('TESTING DATA:')
        self.model.evaluate(self.imagetest, self.labeltest, batch_size=32, verbose=2)

    def save_model(self, model_path='model.pkl'):
        # Save the trained model as a .pkl file
        with open(model_path, 'wb') as file:
            pickle.dump(self.model, file)
        print(f'Model saved to {model_path}')

if __name__ == '__main__':
    classifier = TuberculosisClassifier('./Normal', './Tuberculosis')
    classifier.load_and_preprocess_data()
    classifier.build_model()
    classifier.train_model()
    classifier.evaluate_model()
    classifier.save_model()
