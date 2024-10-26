# streamlit_spectral_clustering.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

module_path = os.path.abspath('/Mr.ML/Clustering')  
if module_path not in sys.path:
    sys.path.append(module_path)

from Spectral import SpectralClusteringModel

def main():
    st.title("Spectral Clustering")

    # Sidebar for configuration
    st.sidebar.header("Upload Data and Set Parameters")

    # Step 1: Upload CSV File
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Load CSV data into a Pandas DataFrame
        data = pd.read_csv(uploaded_file)
        
        # Step 2: Show Dataset Preview
        st.subheader("Dataset Preview")
        st.write("Shape of the dataset:", data.shape)
        st.dataframe(data.head())

        # Step 3: Select Feature Columns for Clustering
        st.sidebar.subheader("Select Feature Columns")
        feature_columns = st.sidebar.multiselect("Select the columns for clustering:", data.columns)

        if len(feature_columns) == 0:
            st.error("Please select at least one feature column for clustering.")
        else:
            # Step 4: Number of Clusters
            n_clusters = st.sidebar.slider("Number of clusters:", 2, 10, 3)

            # Step 6: Prepare data for clustering
            X = data[feature_columns].values

            # Step 7: Run Clustering Algorithm (Spectral Clustering)
            model = SpectralClusteringModel(n_clusters=n_clusters)

            if st.sidebar.button("Run Spectral Clustering"):
                try:
                    model.fit(X)
                    cluster_labels = model.predict(X)

                    # Add cluster labels to the DataFrame
                    data['Cluster'] = cluster_labels

                    # Display clustering result
                    st.subheader("Clustering Result")
                    st.write(data.head())

                    # Plot Clusters (for 2D data only)
                    if X.shape[1] == 2:
                        fig, ax = plt.subplots()
                        scatter = ax.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', s=50)
                        ax.set_xlabel(feature_columns[0])
                        ax.set_ylabel(feature_columns[1])
                        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
                        ax.add_artist(legend1)
                        st.pyplot(fig)
                    else:
                        st.warning("Plotting is only available for 2D data.")

                    # Step 8: Display Metrics
                    metrics = model.get_metrics(X, cluster_labels)
                    st.subheader("Clustering Metrics")
                    for metric, value in metrics.items():
                        st.write(f"**{metric}:** {value:.4f}")

                except Exception as e:
                    st.error(f"Error during clustering: {e}")

            # Step 9: Prediction
            if st.sidebar.checkbox("Predict new data with Spectral Clustering"):
                st.sidebar.subheader("Enter New Data for Prediction")
                new_data = [st.sidebar.number_input(f"Enter a value for {col}:", value=0.0) for col in feature_columns]
                new_data = np.array(new_data).reshape(1, -1)

                if st.sidebar.button("Predict Cluster"):
                    try:
                        predicted_cluster = model.predict(new_data)
                        st.success(f"The predicted cluster for the new data is: **{predicted_cluster[0]}**")
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
