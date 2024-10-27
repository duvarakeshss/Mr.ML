import streamlit as st

# Set page layout
col1, col2 = st.columns([1, 4])
with col1:
    st.image(r"D:\Repos\Mr.ML\assets\ml.webp", width=50)
  
# Sidebar options
st.sidebar.title("Select Model")
model_type = st.sidebar.radio("Choose a model type:", ("Regression", "Classification", "Clustering","Tuberculosis"))


if model_type == "Regression":
    st.title("Regression Models")
    regression_option = st.selectbox("Select Regression Model:", ("Linear Regression", "Multiple Regression", "Decision Tree Regression", "CNN"))
    
    if regression_option == "Linear Regression":

        from linear import main  
        main()  
        
    elif regression_option == "Multiple Regression":
        from Multiple_regression import main
        main()

    elif regression_option == "Decision Tree Regression":
        from R_decision_tree import main
        main()

    elif regression_option == "CNN":
        from cnn import main
        main()

elif model_type == "Classification":
    st.title("Classification Models")
    classification_option = st.selectbox("Select Classification Model:", ("Logistic","KNN","Decision Tree Classifier", "CNN"))
    
    if classification_option == "Logistic":
        from logistic import main
        main()

    elif classification_option == "KNN":
        from knn_stream import main
        main()     
        
    elif classification_option == "Decision Tree Classifier":
        from C_decision_tree import main
        main()   
    
    elif classification_option == "CNN":
        from Classifycnn import run
        run()
    
elif model_type == "Clustering":
    st.title("Clustering Models")
    clustering_option = st.selectbox("Select Clustering Model:", ("K-Means","GMM","Spectral"))
    
    if clustering_option == "K-Means":
        from K_means import main
        main()
    if clustering_option == "GMM":
        from gmm import main
        main()
        
    if clustering_option == "Spectral":
        from spectral import main
        main()
elif model_type == "Tuberculosis":
    from Predict import main
    main()
