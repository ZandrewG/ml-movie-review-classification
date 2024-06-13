import numpy as np
import pandas as pd
import streamlit as st
import joblib


nb_model = joblib.load('multinomial_nb_model.joblib')
lr_model = joblib.load('log_reg_model.joblib')
# Streamlit app
st.title("Movie Review Classification using 2 Machine Learning Models")
st.write("Authored by Loridge Gacho & Zandrew Garais")
st.write("Classify your movie review here:")
text_input = st.text_area("Input:")

vectorizer = joblib.load('vectorizer.pkl')

# Function to preprocess and predict sentiment
def predict(texts, model):
    # Transform the input text using the loaded vectorizer
    text_counts = vectorizer.transform(texts)

    # Predict using the loaded model
    predictions = model.predict(text_counts)

    return predictions


if st.button("Predict"):
        # Predict with model 1
        st.subheader("Prediction using Naive-Bayes")
        arr = [text_input]
        prediction1 = predict(arr, nb_model)
        st.write("Naive-Bayes Model Says it is:",)
        if prediction1[0] == 1:
            st.write("Positive")
        else:
            st.write("Negative")
        
        # Predict with model 2
        st.subheader("Prediction using Logistic Regression")
        arr2 = [text_input]
        prediction2 = predict(arr2, lr_model)
        st.write("Logistic Regression Model Says it is:")
        if prediction2[0] == 1:
            st.write("Positive")
        else:
            st.write("Negative")