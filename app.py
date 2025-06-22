import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# Load Model and Tokenizer
# -------------------------------
@st.cache_resource
def load_prediction_components():
    model = load_model('next_word_prediction_model_gru.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_prediction_components()
max_sequence_len = model.input_shape[1] + 1  # +1 to match tokenizer input logic

# -------------------------------
# Prediction Function
# -------------------------------
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if not token_list:
        return "Input not recognized by tokenizer."

    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted_probs, axis=1)[0]

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return "Word not found in tokenizer."

# -------------------------------
# Streamlit App UI
# -------------------------------
st.set_page_config(page_title="LSTM Next Word Predictor", layout="centered")
st.title("🧠 Next Word Prediction using LSTM")

st.markdown("""
Enter a partial sentence below, and the model will predict the most likely next word based on its training data.
""")

input_text = st.text_input("🔡 Enter your sentence:", placeholder="e.g., Once upon a")

if st.button("🔮 Predict Next Word"):
    if input_text.strip() == "":
        st.warning("Please enter some text before prediction.")
    else:
        with st.spinner("Predicting..."):
            predicted_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
            st.success(f"**Next predicted word:** `{predicted_word}`")
