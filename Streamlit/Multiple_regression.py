# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split

module_path = os.path.abspath('/workspaces/Mr.ML/Regression')  
if module_path not in sys.path:
    sys.path.append(module_path)
from MultipleRegression import MultipleRegression  

def main():
    st.markdown(
    """
    <style>
    .main-title {
        color: #00607a;
        font-size: 32px;
        font-weight: bold;
        text-align: center;
    }
    .section-title {
        color: #00303d;
        font-size: 24px;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .prediction-section {
        background-color: #002029;
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
    .stButton>button {
        background-color: #0c3d37;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #00607a;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    st.markdown('<h1 class="main-title">Multiple Linear Regression</h1>', unsafe_allow_html=True)

    # Sidebar for file upload
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown('<h2 class="section-title">Dataset:</h2>', unsafe_allow_html=True)
        st.write(df.head())

        columns = df.columns.tolist()
        
        # Sidebar for selecting features and label
        st.sidebar.header("Select Features and Label")
        
        # Allow users to select independent features
        independent_vars = st.sidebar.multiselect("Select the independent features:", columns)

        # Allow user to select a dependent label
        dependent_var = st.sidebar.selectbox("Select the dependent variable (label):", columns)

        if independent_vars and dependent_var:
            X = df[independent_vars].values
            y = df[dependent_var].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Create an instance of the MultipleLinearRegression class
            model = MultipleRegression()
            model.fit(X_train, y_train)

            st.markdown('<h2 class="section-title">Regression Results:</h2>', unsafe_allow_html=True)
            st.write("Coefficients:", model.coefficients[1:])
            st.write("Intercept:", model.intercept)

            y_pred = model.predict(X_test)

            mae, mse, r2 = model.evaluate(y_test, y_pred)

            st.markdown('<h2 class="section-title">Model Performance Metrics:</h2>', unsafe_allow_html=True)
            st.write(f"**Mean Absolute Error (MAE)**: {mae:.2f}")
            st.write(f"**Mean Squared Error (MSE)**: {mse:.2f}")
            st.write(f"**R² Score**: {r2:.2f}")

            # Prediction section in the main layout
            st.markdown('<div class="prediction-section"><h3>Make a Prediction</h3>', unsafe_allow_html=True)

            # Input fields for prediction in the main layout
            input_values = []
            for var in independent_vars:
                input_val = st.number_input(f"Enter value for {var}:")
                input_values.append(input_val)

            if st.button("Predict"):
                prediction = model.predict(np.array([input_values]))  # Predict using the model
                st.write(f"Predicted value of {dependent_var}: {prediction[0]:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
# if __name__ == main:
#     main()