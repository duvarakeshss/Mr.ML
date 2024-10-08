import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

st.markdown('<h1 class="main-title">Decision Tree Regressor with Feature Adjustment</h1>', unsafe_allow_html=True)

st.sidebar.header("Upload your CSV file")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(data.head())
    
    label_encoders = {}
    for column in data.columns:
        if data[column].dtype == 'object':
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column].astype(str))
            label_encoders[column] = le

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
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            regressor = DecisionTreeRegressor(random_state=42)
            regressor.fit(X_train, y_train)
            
            y_pred = regressor.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

        
            st.markdown('<h2 class="section-title">Model Performance Metrics</h2>', unsafe_allow_html=True)
            st.write(f"**Mean Absolute Error (MAE)**: {mae:.2f}")
            st.write(f"**Mean Squared Error (MSE)**: {mse:.2f}")
            st.write(f"**R² Score**: {r2:.2f}")

            st.markdown('<h2 class="section-title">Decision Tree Visualization</h2>', unsafe_allow_html=True)
            fig = plt.figure(figsize=(20, 20))  
            tree.plot_tree(regressor, feature_names=features, filled=True, rounded=True, fontsize=10)
            plt.title("Decision Tree Regressor", fontsize=16)
            st.pyplot(fig)
