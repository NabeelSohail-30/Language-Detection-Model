import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder


def load_saved_model(filename='language_detection_model.sav'):
    loaded_model, loaded_vectorizer, loaded_label_encoder = joblib.load(
        filename)
    return loaded_model


def predict_language(model, user_input):
    languages = {'0': 'English', '1': 'French', '2': 'Spanish'}

    # Predict the language label
    predicted_label = model.predict([user_input])

    # Map label to language
    predicted_language = languages.get(
        str(predicted_label[0]), "Not Predicted")

    return predicted_language


def main():
    # Load the model and related components
    model = load_saved_model()

    # Streamlit UI
    st.title("Language Detection Web App")

    # User input
    user_input = st.text_input("Enter a sentence:")

    if st.button("Predict Language"):
        if user_input:
            # Predict language
            predicted_language = predict_language(
                model, user_input)

            # Display result
            st.success(f"Predicted Language: {predicted_language}")
        else:
            st.warning("Please enter a sentence for prediction.")


if __name__ == "__main__":
    main()
