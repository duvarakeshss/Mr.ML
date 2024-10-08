import streamlit as st
import pandas as pd
import numpy as np

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

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('<h2 class="section-title">Dataset:</h2>', unsafe_allow_html=True)
    st.write(df.head())

    columns = df.columns.tolist()
    independent_vars = st.multiselect(
        "Select the independent features:", columns
    )

    dependent_var = [col for col in columns if col not in independent_vars]

    if independent_vars and dependent_var:
        X = df[independent_vars].values
        y = df[dependent_var[0]].values

        X = np.column_stack((np.ones(X.shape[0]), X))

        X_transpose = X.T
        coefficients = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)

        st.markdown('<h2 class="section-title">Regression Results:</h2>', unsafe_allow_html=True)
        st.write("Coefficients:", coefficients[1:])
        st.write("Intercept:", coefficients[0])

        st.markdown('<div class="prediction-section"><h3>Make a Prediction</h3>', unsafe_allow_html=True)
        input_values = []
        for var in independent_vars:
            input_val = st.number_input(f"Enter value for {var}:")
            input_values.append(input_val)

        if st.button("Predict"):
            prediction = coefficients[0] + np.dot(coefficients[1:], input_values)
            st.write(f"Predicted value of {dependent_var[0]}: {prediction}")
        st.markdown('</div>', unsafe_allow_html=True)
