import streamlit as st
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="😊")

@st.cache_resource
def download_nltk():
    nltk.download("stopwords")
    nltk.download("punkt")

download_nltk()

@st.cache_data
def load_data():
    train = pd.read_csv("twitter_training.csv", header=None)
    val = pd.read_csv("twitter_validation.csv", header=None)

    train.columns = ["id", "info", "type", "text"]
    val.columns = ["id", "info", "type", "text"]

    train = train.dropna(subset=["type", "text"])
    val = val.dropna(subset=["type", "text"])

    train["type"] = train["type"].str.strip().str.lower()
    val["type"] = val["type"].str.strip().str.lower()

    return train, val

train_data, val_data = load_data()

def clean_text(text):
    text = text.lower()
    return re.sub(r"[^a-z0-9 ]+", " ", text)

train_data["clean_text"] = train_data["text"].apply(clean_text)
val_data["clean_text"] = val_data["text"].apply(clean_text)

vectorizer = CountVectorizer(
    stop_words=nltk.corpus.stopwords.words("english"),
    max_features=3000
)

X_train = vectorizer.fit_transform(train_data["clean_text"])
X_val = vectorizer.transform(val_data["clean_text"])

le = LabelEncoder()
y_train = le.fit_transform(train_data["type"])
y_val = le.transform(val_data["type"])

model = LogisticRegression(
    solver="liblinear",
    class_weight="balanced",
    max_iter=500
)

model.fit(X_train, y_train)

preds = model.predict(X_val)
accuracy = accuracy_score(y_val, preds)

st.title("📊 Twitter Sentiment Analysis")
st.write("Model: Logistic Regression")
st.metric("Validation Accuracy", f"{accuracy*100:.2f}%")

text = st.text_area("Enter a tweet")

if st.button("Analyze"):
    if text.strip():
        vec = vectorizer.transform([clean_text(text)])
        pred = model.predict(vec)[0]
        st.success(le.inverse_transform([pred])[0])
