# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

module_path = os.path.abspath('D:/Repos/Mr.ML/Clustering')  
if module_path not in sys.path:
    sys.path.append(module_path)
from GMM import GaussianClustering

# Set page layout to wide
st.set_page_config(layout="wide")

def main():
    st.title("Gaussian Mixture Model")

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
            n_clusters = st.sidebar.slider("Number of clusters (components):", 2, 10, 3)

            # Step 5: Prepare data for GMM
            X = data[feature_columns].values

            # Step 6: Initialize and Fit GMM
            gmm = GaussianClustering(n_components=n_clusters)
            
            if st.sidebar.button("Run GMM Clustering"):
                try:
                    gmm.fit(X)
                    cluster_labels = gmm.predict(X)

                    # Add cluster labels to the DataFrame
                    data['Cluster'] = cluster_labels

                    # Display clustering result
                    st.subheader("Clustering Result")
                    st.write(data.head())

                    # Step 7: Plot Clusters (for 2D data only)
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
                        
                except Exception as e:
                    st.error(f"Error during GMM clustering: {e}")

if __name__ == "__main__":
    main()
