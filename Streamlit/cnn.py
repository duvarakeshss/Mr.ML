import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Adding the path for the custom CNN model
module_path = os.path.abspath('/workspaces/Mr.ML/Regression')  
if module_path not in sys.path:
    sys.path.append(module_path)

# Import the CNN class
from Cnn import CNN



def main():
    st.title("CNN Model for Regression")

    # Sidebar for file upload and inputs
    st.sidebar.header("Upload Data and Set Parameters")
    
    # Step 1: File upload
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Read the uploaded CSV
        data = pd.read_csv(uploaded_file)

        # Step 2: Show the dataset
        st.subheader("Dataset Preview")
        st.dataframe(data.head())

        # Step 3: Select Feature and Label Columns
        st.sidebar.subheader("Select Feature and Label")
        feature_columns = st.sidebar.multiselect("Select the feature columns:", data.columns)
        label_column = st.sidebar.selectbox("Select the label (target) column:", data.columns)

        # Step 4: Get CNN Hyperparameters
        learning_rate = st.sidebar.slider("Select the learning rate:", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001)
        epochs = st.sidebar.slider("Select the number of epochs:", min_value=1, max_value=100, value=10, step=1)

        # Step 5: Train the model and display results when the user clicks "Run"
        if st.sidebar.button("Run CNN"):
            if len(feature_columns) == 0 or not label_column:
                st.error("Please select both features and label columns.")
            else:
                # Prepare the data for CNN
                X = data[feature_columns].values
                y = data[label_column].values

                # Scale features and labels
                scaler_X = MinMaxScaler()
                scaler_y = MinMaxScaler()

                X_scaled = scaler_X.fit_transform(X)
                y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

                # Reshape the data for CNN
                X_scaled = X_scaled.reshape(-1, len(feature_columns), 1)

                # Check shapes
                st.write("Features shape after reshaping:", X_scaled.shape)
                st.write("Labels shape:", y_scaled.shape)

                # Train-test split (80% train, 20% test)
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

                # Initialize and train the CNN
                cnn = CNN(learning_rate=learning_rate)
                
                with st.spinner("Training the CNN..."):
                    try:
                        cnn.train(X_train, y_train, epochs=epochs)

                        # Evaluate the model
                        loss = cnn.evaluate(X_test, y_test)

                        st.subheader("Model Results")
                        st.write(f"**Final Loss on Test Set:** {loss:.4f}")

                        # Plot the training loss
                        st.subheader("Training Loss Over Epochs")
                        st.line_chart(cnn.loss_history)

                        # Store the trained model in session state
                        st.session_state['cnn'] = cnn
                        st.session_state['scaler_y'] = scaler_y  # Store the scaler for later use
                    except Exception as e:
                        st.error(f"Error during training: {e}")

        # Step 6: Predict values using the trained model (only if a model is available)
        if 'cnn' in st.session_state:
            cnn = st.session_state['cnn']
            scaler_y = st.session_state['scaler_y']  # Get the scaler

            st.sidebar.subheader("Make Predictions")
            input_values = [st.sidebar.number_input(f"Enter a value for {col}:", value=0.0) for col in feature_columns]
            input_values = np.array(input_values).reshape(-1, len(feature_columns), 1)

            if st.sidebar.button("Predict"):
                try:
                    # Make predictions using the trained CNN model
                    predicted_value_scaled = cnn.predict(input_values)

                    # Inverse transform the prediction
                    predicted_value = scaler_y.inverse_transform(predicted_value_scaled)
                    
                    st.session_state['predicted_value'] = predicted_value
                    st.success(f"Predicted value for {label_column}: **{predicted_value[0][0]:.4f}**")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

        # Display the prediction result if available
        if 'predicted_value' in st.session_state:
            st.subheader("Prediction Result")
            st.write(f"Predicted value for {label_column}: **{st.session_state['predicted_value'][0][0]:.4f}**")

# if __name__ == "__main__":
#     main()
