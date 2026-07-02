import streamlit as st
import joblib
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Get the folder this script lives in — works on any machine
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

nltk.download('stopwords', quiet=True)

model = joblib.load(os.path.join(BASE_DIR, 'fake_news_model.joblib'))
vectorizer = joblib.load(os.path.join(BASE_DIR, 'vectorizer.joblib'))

port_stem = PorterStemmer()
stop_words = set(stopwords.words('english'))

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stop_words]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
st.markdown("<h1 style='text-align:center'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray'>Paste any American news article to check if its real or fake</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
col1.metric("Test Accuracy", "98.6%")
col2.metric("Trained On", "44K articles")
col3.metric("Model", "Logistic Reg.")

st.divider()

news_input = st.text_area("Paste reutor or American news article here:", height=180, placeholder="Enter article text...")

if st.button("Analyze Article", type="primary", use_container_width=True):
    if news_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            cleaned = stemming(news_input)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)
            probability = model.predict_proba(vectorized)[0]
        if prediction[0] == 1:
            confidence = round(probability[1] * 100, 1)
            st.success(f"✅ Real News — {confidence}% confidence")
        else:
            confidence = round(probability[0] * 100, 1)
            st.error(f"❌ Fake News — {confidence}% confidence")