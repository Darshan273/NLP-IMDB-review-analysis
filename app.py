import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

st.title("🎬 Movie Review Sentiment Analysis")

review = st.text_area("Enter your review here:")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter some text")
    else:
        clean_review = clean_text(review)
        vector = vectorizer.transform([clean_review])
        prediction = model.predict(vector)
        st.success(f"Sentiment: **{prediction[0]}**")
