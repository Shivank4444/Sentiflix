# Step 1: Import Libraries
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.datasets import imdb # type: ignore
from tensorflow.keras.preprocessing import sequence # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import streamlit as st

# Load RNN model
model = load_model('simple_rnn_imdb.h5')


# 🔥 Load Logistic Regression model
logistic_model = joblib.load('logistic_model_imdb.pkl')

# Word index for preprocessing RNN input
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Helper Functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Streamlit UI
st.markdown('#  <span class="sentiflix-title">|</span> SentiFlix', unsafe_allow_html=True)
st.markdown("""
        <style>
        .stButton > button {
            background-color: black !important;
            color: #F5C518 !important;
            font-weight: bold;
            border: 2px solid #F5C518 !important;
            transition: color 0.3s ease, border-color 0.3s ease !important;
        }
        .stButton > button:hover {
            background-color: black !important;
            color: white !important;
            border: 2px solid white !important;
        }
        .stButton > button:active {
            background-color: black !important;
            color: white !important;
            border: 2px solid white !important;
        }
        .imdb-title {
            color: #F5C518;
            font-weight: bold;
        }
        .sentiflix-title{
            color: #F5C518;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)
st.markdown("# 🎬<span class='imdb-title'>IMDb</span> Movie Review Sentiment Analysis", unsafe_allow_html=True)
st.write("Enter a movie review to classify it as positive or negative.")

user_input = st.text_area('Movie Review')

if st.button('Classify'):
    # --- RNN Prediction ---
    preprocessed_input = preprocess_text(user_input)
    prediction = model.predict(preprocessed_input)
    rnn_sentiment = 'Positive 👍' if prediction[0][0] > 0.5 else 'Negative 👎'

    # --- Logistic Regression Prediction ---
    logistic_pred = logistic_model.predict([user_input])[0]
    logistic_proba = logistic_model.predict_proba([user_input])[0]
    logistic_sentiment = 'Positive 👍' if logistic_pred == 1 else 'Negative 👎'

    # --- Display Results ---
    st.subheader("🔁 RNN Prediction:")
    st.write(f"Sentiment: {rnn_sentiment}")
    st.write(f"Prediction Score: {prediction[0][0]:.4f}")

    st.subheader("📊 Logistic Regression Prediction:")
    st.write(f"Sentiment: {logistic_sentiment}")
    st.write(f"Probability (Positive): {logistic_proba[1]:.4f}")
else:
    st.write("Please enter a movie review.")
