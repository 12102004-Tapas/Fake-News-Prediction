# 📰 Fake News Detector

A machine learning web app that classifies news articles as **Real** or **Fake** using NLP techniques.

## 🔍 Overview
This project uses Logistic Regression trained on the ISOT Fake News dataset (~44,000 articles) to detect fake news with **98.6% test accuracy**. Text is cleaned and preprocessed using NLTK, then vectorized with TF-IDF before classification.

## 📊 Dataset
- **Source:** ISOT Fake News Dataset (Kaggle)
- Two CSV files: `True.csv` (real news) and `Fake.csv` (fake news), combined and labeled during preprocessing
- ~44,000 articles total

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
   git clone https://github.com/12102004-Tapas/your-repo-name.git
   cd your-repo-name
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
- `fake_news_model.joblib` — Trained Logistic Regression model
- `vectorizer.joblib` — TF-IDF vectorizer

## 🎥 Demo
[Add your screen recording link or embed here]

## 👤 Author
Tapas Mahapatra — MCA Student, Data Science & Analytics
