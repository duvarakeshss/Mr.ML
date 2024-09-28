import streamlit as st
import pandas as pd
import sys
import os

# Adjusting system path to ensure the module can be imported
module_path = os.path.abspath('D:/Repos/Mr.ML/Regression')  # Correct absolute path to Regression folder
if module_path not in sys.path:
    sys.path.append(module_path)

# Import the LinearRegressionModel after adjusting the path
from LinearRegression import LinearRegressionModel

def main():
    st.title("Linear Regression Model")

    # Step 1: File upload
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        # Read the uploaded CSV (handle as a file-like object)
        data = pd.read_csv(uploaded_file)

        # Step 2: Show the dataset
        st.write("### Dataset Preview")
        st.write(data.head())

        # Step 3: Select Feature and Label Columns
        st.write("### Select Feature and Label")
        feature = st.selectbox("Select the feature column:", data.columns)
        label = st.selectbox("Select the label (target) column:", data.columns)

        # Step 4: Get Learning Rate
        learning_rate = st.number_input("Enter the learning rate (e.g., 0.01):", value=0.01, min_value=0.001)

        # Step 5: Train the model and display results when the user clicks "Run"
        if st.button("Run Linear Regression"):
            model = LinearRegressionModel(data, learning_rate)
            model.trainAndTestModel(feature, label)

            # Display metrics
            st.write(f"### Equation of line: y = {model.m:.4f}x + {model.b:.4f}")
            st.write(f"Mean Squared Error: {model.mse}")
            st.write(f"Mean Absolute Error: {model.mae}")
            st.write(f"R Squared Value: {model.r2}")

            # Plot the regression line
            st.write("### Regression Plot")
            st.pyplot(model.fig)

            # Store the model in session state
            st.session_state['model'] = model

    # Step 6: Predict values using the trained model (only if a model is available)
    if 'model' in st.session_state:
        model = st.session_state['model']
        
        st.write("### Make Predictions")
        input_value = st.number_input(f"Enter a value for {feature}:")
        
        if st.button("Predict"):
            predicted_value = model.m * input_value + model.b
            st.session_state['predicted_value'] = predicted_value

    # Display the prediction result if available
    if 'predicted_value' in st.session_state:
        st.write(f"Predicted value for {label}: {st.session_state['predicted_value']:.4f}")

if __name__ == "__main__":
    main()
