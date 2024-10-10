import streamlit as st
import pandas as pd
import sys
import os

module_path = os.path.abspath('/workspaces/Mr.ML/Regression')  
if module_path not in sys.path:
    sys.path.append(module_path)

# Import the LinearRegressionModel
from LinearRegression import LinearRegressionModel

def main():
    st.title("Linear Regression Model")

    # Sidebar for file upload and inputs
    st.sidebar.header("Upload Data and Set Parameters")
    
    # Step 1: File upload
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Read the uploaded CSV (handle as a file-like object)
        data = pd.read_csv(uploaded_file)

        # Step 2: Show the dataset
        st.subheader("Dataset Preview")
        st.dataframe(data.head())

        # Step 3: Select Feature and Label Columns
        st.sidebar.subheader("Select Feature and Label")
        feature = st.sidebar.selectbox("Select the feature column:", data.columns)
        label = st.sidebar.selectbox("Select the label (target) column:", data.columns)

        # Step 4: Get Learning Rate
        learning_rate = st.sidebar.slider("Select the learning rate:", min_value=0.001, max_value=0.1, value=0.01, step=0.001)

        # Step 5: Train the model and display results when the user clicks "Run"
        if st.sidebar.button("Run Linear Regression"):
            model = LinearRegressionModel(data, learning_rate)
            model.trainAndTestModel(feature, label)

            # Display metrics in the main panel
            st.subheader("Model Results")
            st.write(f"**Equation of line:** y = {model.m:.4f}x + {model.b:.4f}")
            st.write(f"**Mean Squared Error (MSE):** {model.mse}")
            st.write(f"**Mean Absolute Error (MAE):** {model.mae}")
            st.write(f"**R Squared Value (R²):** {model.r2}")

            # Plot the regression line
            st.subheader("Regression Plot")
            st.pyplot(model.fig)

            # Store the model in session state
            st.session_state['model'] = model

    # Step 6: Predict values using the trained model (only if a model is available)
    if 'model' in st.session_state:
        model = st.session_state['model']
        
        st.sidebar.subheader("Make Predictions")
        input_value = st.sidebar.number_input(f"Enter a value for {feature}:")
        
        if st.sidebar.button("Predict"):
            predicted_value = model.m * input_value + model.b
            st.session_state['predicted_value'] = predicted_value

    # Display the prediction result if available
    if 'predicted_value' in st.session_state:
        st.subheader("Prediction Result")
        st.write(f"Predicted value for {label}: **{st.session_state['predicted_value']:.4f}**")

# if __name__ == "__main__":
#     main()
