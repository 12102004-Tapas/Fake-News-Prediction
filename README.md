# 📰 Fake News Detector

A machine learning web app that classifies news articles as **Real** or **Fake** using NLP techniques.

## 🔍 Overview
This project uses Logistic Regression trained on the ISOT Fake News dataset (~44,000 articles) to detect fake news with **98.6% test accuracy**. Text is cleaned and preprocessed using NLTK, then vectorized with TF-IDF before classification.

## 📊 Dataset
- **Source:** ISOT Fake News Dataset (Kaggle)
- `True.csv` — real news articles
- `Fake.csv` — fake news articles
- ~44,000 articles total, combined and labeled during preprocessing

## 🛠️ Tech Stack
- **Model:** Logistic Regression
- **Vectorization:** TF-IDF
- **Preprocessing:** NLTK (stopword removal, Porter stemming)
- **Frontend:** Streamlit

## 📊 Performance
- Test Accuracy: 98.6%
- Trained on: 44,000 articles

## 🚀 How to Run Locally
1. Clone the repo:
```bash
   git clone https://github.com/12102004-Tapas/Fake-News-Prediction.git
   cd Fake-News-Prediction
```
2. Install dependencies:
```bash
   pip install streamlit joblib scikit-learn nltk
```
3. Run the app:
```bash
   streamlit run app.py
```

## 📁 Files
- `app.py` — Streamlit app code
- `FakeNewsPrediction.ipynb` — Model training notebook
- `fake_news_model.joblib` — Trained Logistic Regression model
- `vectorizer.joblib` — TF-IDF vectorizer
- `True.csv`, `Fake.csv` — Dataset (ISOT)

## 🎥 Demo
[Add your screen recording or GIF here]

## 👤 Author
Tapas Mahapatra — MCA Student, Data Science & Analytics
