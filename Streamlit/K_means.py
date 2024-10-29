# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Set up module path for KMeans class import
module_path = os.path.abspath('D:/Repos/Mr.ML/Clustering')  
if module_path not in sys.path:
    sys.path.append(module_path)
from kmeans import KMeans

def main():
    # Streamlit app title and styling
    st.markdown("""
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
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-title">K-Means Clustering</h1>', unsafe_allow_html=True)

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        data = KMeans.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(pd.DataFrame(data))

        # Input for number of clusters
        k = st.number_input("Number of clusters (k):", min_value=1, max_value=10, value=3)

        if st.button("Run K-Means"):
            kmeans = KMeans(k)
            centroids, clusters = kmeans.fit(data)

            # Display results
            st.markdown('<h2 class="section-title">Results</h2>', unsafe_allow_html=True)
            st.write("Final Centroids:")
            for idx, centroid in enumerate(centroids):
                st.write(f"Centroid {idx + 1}: {centroid}")

            st.write("\nCluster Assignments:")
            for idx, cluster in enumerate(clusters):
                st.write(f"Cluster {idx + 1}: {len(cluster)} points")

            # Plot clusters
            plt.figure(figsize=(10, 6))
            colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'black', 'orange', 'purple', 'brown']
            for idx, cluster in enumerate(clusters):
                if cluster:  # If the cluster is not empty
                    cluster = np.array(cluster)
                    plt.scatter(cluster[:, 0], cluster[:, 1], color=colors[idx % len(colors)], label=f'Cluster {idx + 1}')
            plt.scatter(np.array(centroids)[:, 0], np.array(centroids)[:, 1], color='black', marker='X', s=200, label='Centroids')
            plt.title("K-Means Clustering Result")
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.legend()

            # Show the plot in Streamlit
            st.pyplot(plt)

            # Clear the figure to prevent overlap in subsequent runs
            plt.clf()