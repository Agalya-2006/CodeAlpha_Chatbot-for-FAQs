import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("FAQ Chatbot 🤖")

data = pd.read_csv("faq.csv")

questions = data["question"]
answers = data["answer"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

user_input = st.text_input("Ask a question:")

if user_input:
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    index = similarity.argmax()
    st.write("Chatbot:", answers[index])