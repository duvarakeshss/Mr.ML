import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os

# Adding the path for the custom Logistic Regression model
module_path = os.path.abspath('D:/Repos/Mr.ML/Classification')  
if module_path not in sys.path:
    sys.path.append(module_path)

from logisiticRegression import LogisticRegression

def main():
    st.title("Logistic Regression")

    # Sidebar for file upload
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(data.head())

        # Select features and label column in the sidebar
        st.sidebar.header("Select Features and Label")
        feature_columns = st.sidebar.multiselect("Select feature columns", data.columns.tolist())
        label_column = st.sidebar.selectbox("Select label (target) column", data.columns.tolist())

        if st.sidebar.button("Train Model"):
            if len(feature_columns) == 0 or not label_column:
                st.error("Please select both feature and label columns.")
            else:
                X = data[feature_columns].values
                y = data[label_column].values

                # Split the dataset into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Normalize the features
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # Train the model
                model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
                model.fit(X_train, y_train)

                # Store the model and scaler in session state
                st.session_state['model'] = model
                st.session_state['scaler'] = scaler
                st.session_state['feature_columns'] = feature_columns
                st.session_state['label_column'] = label_column

                # Make predictions and evaluate the model
                predictions = model.predict(X_test)
                accuracy = model.accuracy(y_test, predictions)
                st.success(f'Test accuracy: {accuracy:.4f}')

        # Sidebar for user prediction input
        if 'model' in st.session_state:
            st.sidebar.header("Make a Prediction")
            user_input = []
            for feature in st.session_state['feature_columns']:
                value = st.sidebar.number_input(f"Enter value for {feature}", value=0.0)
                user_input.append(value)

            if st.sidebar.button("Predict"):
                user_input_scaled = st.session_state['scaler'].transform([user_input])  # Scale the input
                user_prediction = st.session_state['model'].predict(user_input_scaled)
                st.success(f'The predicted class for the input values is: {user_prediction[0]}')

if __name__ == "__main__":
    main()
