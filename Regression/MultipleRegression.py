import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Custom CSS styles
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

# Title of the app
st.markdown('<h1 class="main-title">Multiple Linear Regression</h1>', unsafe_allow_html=True)

# File uploader for CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('<h2 class="section-title">Dataset:</h2>', unsafe_allow_html=True)
    st.write(df.head())

    columns = df.columns.tolist()
    independent_vars = st.multiselect("Select the independent features:", columns)

    dependent_var = [col for col in columns if col not in independent_vars]

    if independent_vars and dependent_var:
        # Prepare the data
        X = df[independent_vars].values
        y = df[dependent_var[0]].values

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Add bias term (intercept) for linear regression
        X_train = np.column_stack((np.ones(X_train.shape[0]), X_train))
        X_test = np.column_stack((np.ones(X_test.shape[0]), X_test))

        # Calculate coefficients using the Normal Equation
        X_transpose = X_train.T
        coefficients = np.linalg.inv(X_transpose.dot(X_train)).dot(X_transpose).dot(y_train)

        st.markdown('<h2 class="section-title">Regression Results:</h2>', unsafe_allow_html=True)
        st.write("Coefficients:", coefficients[1:])
        st.write("Intercept:", coefficients[0])

        # Make predictions on the test set
        y_pred = X_test.dot(coefficients)

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Display metrics
        st.markdown('<h2 class="section-title">Model Performance Metrics:</h2>', unsafe_allow_html=True)
        st.write(f"**Mean Absolute Error (MAE)**: {mae:.2f}")
        st.write(f"**Mean Squared Error (MSE)**: {mse:.2f}")
        st.write(f"**R² Score**: {r2:.2f}")

        # Prediction Section
        st.markdown('<div class="prediction-section"><h3>Make a Prediction</h3>', unsafe_allow_html=True)
        input_values = []
        for var in independent_vars:
            input_val = st.number_input(f"Enter value for {var}:")
            input_values.append(input_val)

        if st.button("Predict"):
            prediction = coefficients[0] + np.dot(coefficients[1:], input_values)
            st.write(f"Predicted value of {dependent_var[0]}: {prediction:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
