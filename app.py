import streamlit as st
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="😊",
    layout="centered"
)

# -----------------------------
# Download NLTK data
# -----------------------------
@st.cache_resource
def download_nltk():
    nltk.download("stopwords")
    nltk.download("punkt")

download_nltk()

# -----------------------------
# Load & clean data
# -----------------------------
@st.cache_data
def load_data():
    train = pd.read_csv("twitter_training.csv", header=None)
    val = pd.read_csv("twitter_validation.csv", header=None)

    train.columns = ["id", "info", "type", "text"]
    val.columns = ["id", "info", "type", "text"]

    # Remove bad rows
    train = train.dropna(subset=["type", "text"])
    val = val.dropna(subset=["type", "text"])

    # Normalize labels
    train["type"] = train["type"].str.strip().str.lower()
    val["type"] = val["type"].str.strip().str.lower()

    return train, val

train_data, val_data = load_data()

# -----------------------------
# Text cleaning
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return text

train_data["clean_text"] = train_data["text"].apply(clean_text)
val_data["clean_text"] = val_data["text"].apply(clean_text)

# -----------------------------
# Vectorization
# -----------------------------
vectorizer = CountVectorizer(
    stop_words=nltk.corpus.stopwords.words("english"),
    max_features=5000
)

X_train = vectorizer.fit_transform(train_data["clean_text"])
X_val = vectorizer.transform(val_data["clean_text"])

# -----------------------------
# Label Encoding (CRITICAL)
# -----------------------------
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_data["type"])
y_val = label_encoder.transform(val_data["type"])

# -----------------------------
# Logistic Regression (PROPER)
# -----------------------------
model = LogisticRegression(
    max_iter=1000,
    solver="lbfgs",
    class_weight="balanced",
    multi_class="auto",
    n_jobs=-1
)

model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
val_preds = model.predict(X_val)
accuracy = accuracy_score(y_val, val_preds)

# -----------------------------
# UI
# -----------------------------
st.title("📊 Twitter Sentiment Analysis")
st.caption("Model used: **Logistic Regression**")

st.markdown("---")

user_input = st.text_area("✍️ Enter a tweet:")

if st.button("🚀 Analyze Sentiment"):
    if user_input.strip():
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        label = label_encoder.inverse_transform([pred])[0]
        st.success(f"✅ **Predicted Sentiment:** {label}")
    else:
        st.warning("Please enter text")

st.markdown("---")
st.metric("Validation Accuracy", f"{accuracy * 100:.2f}%")
