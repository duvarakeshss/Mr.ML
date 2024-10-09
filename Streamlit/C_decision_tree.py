# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from sklearn import tree

module_path = os.path.abspath('D:/Repos/Mr.ML/Classification')  
if module_path not in sys.path:
    sys.path.append(module_path)
from decision_tree import DecisionTreeModel


def main():
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
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-title">Decision Tree Classifier with Feature Adjustment</h1>', unsafe_allow_html=True)

    st.sidebar.header("Upload your CSV file")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(data.head())
        
        dt_model = DecisionTreeModel()
        data = dt_model.encode_labels(data)

        st.sidebar.header("Select Features and Target")
        
        target_variable = st.sidebar.selectbox('Select Target Variable:', data.columns)
        
        features = st.sidebar.multiselect('Select Features:', data.columns.difference([target_variable]))

        if features and target_variable:
            X = data[features]
            y = data[target_variable]

            st.sidebar.header("Adjust Feature Values")
            for feature in features:
                min_value = float(data[feature].min())
                max_value = float(data[feature].max())
                step = (max_value - min_value) / 100
                value = st.sidebar.slider(f'Adjust {feature}', min_value, max_value, (min_value + max_value) / 2, step)

            if st.sidebar.button("Train Decision Tree"):
                X_test, y_test = dt_model.train(X, y)

                st.markdown('<h2 class="section-title">Decision Tree Visualization</h2>', unsafe_allow_html=True)
                fig = plt.figure(figsize=(20, 20))  
                tree.plot_tree(dt_model.get_tree(), feature_names=features, class_names=dt_model.label_encoders[target_variable].classes_, filled=True, rounded=True, fontsize=10)
                plt.title("Decision Tree Classifier", fontsize=16)
                st.pyplot(fig)
