import streamlit as st
import pandas as pd
import sys
import os

module_path = os.path.abspath('D:/Repos/Mr.ML/Classification')  
if module_path not in sys.path:
    sys.path.append(module_path)
from knn import Knn_Classification

# Streamlit app
st.title("K-Nearest Neighbors Classifier with Multi-Feature Selection")

# Sidebar for CSV upload
st.sidebar.header("Upload CSV Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Display the dataset
    st.write("Uploaded Dataset:")
    st.write(data)

    # Select multiple feature columns (independent variables)
    selected_features = st.sidebar.multiselect("Select feature columns (inputs):", data.columns[:-1])
    
    # Select label column (dependent variable)
    selected_label = st.sidebar.selectbox("Select the label column (target):", data.columns)

    if selected_features and selected_label:

        # Select k value
        k = st.sidebar.slider('Select k value for KNN:', 1, 15, value=3)

        # Create and train the KNN classifier
        knn_classifier = Knn_Classification(k, data, selected_features, selected_label)

        # Evaluate the model
        accuracy, conf_matrix, report = knn_classifier.evaluate()

        # Display the evaluation results
        st.subheader("Model Performance")
        st.write(f"Accuracy: {accuracy * 100:.2f}%")

        st.subheader("Confusion Matrix")
        st.write(conf_matrix)

        st.subheader("Classification Report")
        st.text(report)

        # Allow user to input new data for prediction based on selected features
        st.sidebar.subheader("Prediction Input")
        input_data = {}
        for feature in selected_features:
            input_data[feature] = st.sidebar.number_input(f"Enter value for {feature}:", value=float(data[feature].mean()))
        
        if st.sidebar.button("Predict"):
            input_df = pd.DataFrame([input_data])
            prediction = knn_classifier.predict(input_df)
            st.subheader("Prediction")
            st.write(f"Predicted class for input data: {prediction[0]}")
    else:
        st.error("Please select both features and target to proceed.")
else:
    st.write("Please upload a CSV file to get started.")
