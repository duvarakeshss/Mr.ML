import streamlit as st
import pickle
import numpy as np
import cv2 as cv
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

class ModelPredictor:
    def __init__(self, model_path='model.pkl'):
        # Load the trained model
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)

    def preprocess_image(self, image, image_size=256):
        # Preprocess the image for prediction
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.resize(image, (image_size, image_size))
        image = image.astype('float32') / 255
        image = image.reshape(1, image_size, image_size, 1)
        return image

    def predict(self, image):
        # Make a prediction
        processed_image = self.preprocess_image(image)
        prediction = self.model.predict(processed_image)
        label = 'Tuberculosis' if prediction > 0.5 else 'Normal'
        return label, prediction

# Streamlit app
def main():
    st.title("Tuberculosis Detection")

    # Sidebar for image upload
    st.sidebar.title("Options")
    st.sidebar.write("Upload a chest X-ray image to detect Tuberculosis")

    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv.imdecode(file_bytes, 1)

        # Display the image with a reduced size (e.g., width=300)
        st.image(image, caption="Uploaded Image", use_column_width='auto', width=300)

        # Predict the uploaded image
        st.write("Classifying...")
        predictor = ModelPredictor('model.pkl')
        label, prediction = predictor.predict(image)

        # Display the prediction result
        st.write(f"The prediction is: {label}")
        st.write(f"Confidence: {prediction[0][0]:.2f}")

        # Optionally show evaluation metrics if true labels are available
        if 'imagetest' in st.session_state and 'labeltest' in st.session_state:
            imagetest = st.session_state['imagetest']
            labeltest = st.session_state['labeltest']
            predictions = (predictor.model.predict(imagetest) > 0.5).astype('int32')
            
            # Show evaluation metrics
            st.write("## Evaluation Metrics")
            st.text("Classification Report")
            st.text(classification_report(labeltest, predictions))
            st.text("Confusion Matrix")
            st.text(confusion_matrix(labeltest, predictions))

if __name__ == '__main__':
    main()
