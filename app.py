import streamlit as st
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set Streamlit page configuration
st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="😊", layout="centered")

# Download necessary NLTK resources
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords')
    nltk.download('punkt')

download_nltk_data()

# Load datasets
@st.cache_data
def load_data():
    train = pd.read_csv("twitter_training.csv", header=None)
    val = pd.read_csv("twitter_validation.csv", header=None)
    
    train.columns = ['id', 'information', 'type', 'text']
    val.columns = ['id', 'information', 'type', 'text']

    #  FIX: remove rows with missing labels
    train = train.dropna(subset=["type"])
    val = val.dropna(subset=["type"])

    # FIX: normalize label text
    train["type"] = train["type"].str.strip().str.lower()
    val["type"] = val["type"].str.strip().str.lower()
    
    return train, val

train_data, val_data = load_data()

# Text Preprocessing
def clean_text(text):
    if isinstance(text, str):  # Ensure only strings are processed
        text = text.lower()
        text = re.sub(r'[^A-Za-z0-9 ]+', ' ', text)
        return text
    return ""  # Return empty string for non-string values

@st.cache_data
def preprocess_data(train_df, val_df):
    train_df["clean_text"] = train_df["text"].fillna(" ").apply(clean_text)
    val_df["clean_text"] = val_df["text"].fillna(" ").apply(clean_text)

    vectorizer = CountVectorizer(
        stop_words=nltk.corpus.stopwords.words('english'),
        max_features=5000
    )

    X_train_bow = vectorizer.fit_transform(train_df["clean_text"])
    X_val_bow = vectorizer.transform(val_df["clean_text"])
    
    return vectorizer, X_train_bow, X_val_bow


vectorizer, X_train_bow, X_val_bow = preprocess_data(train_data, val_data)


# Train Model (Removed Caching to Fix UnhashableParamError)
def train_model(X_train, y_train):
    
    model = LogisticRegression(C=1, solver="liblinear", max_iter=200)
    model.fit(X_train, y_train)
    return model


model = train_model(X_train_bow, train_data["type"])

@st.cache_data
def compute_validation_accuracy():
    y_val = val_data["type"]  # 🔧 FIX: no string casting
    val_predictions = model.predict(X_val_bow)
    return accuracy_score(y_val, val_predictions)


validation_accuracy = compute_validation_accuracy()

# Streamlit UI with Enhanced Design
st.markdown("""
    <style>
        .main { background-color: #f0f2f6; }
        h1 { color: #333366; text-align: center; }
        .stTextArea>label { font-size: 18px; font-weight: bold; }
        .stButton>button { background-color: #333366; color: white; border-radius: 10px; }
        .stButton>button:hover { background-color: #555599; }
        .stSuccess { font-size: 20px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Twitter Sentiment Analysis")
st.subheader("🔍 Analyze the sentiment of tweets using a trained ML model!")

st.markdown("""
    This application takes in a tweet and predicts its sentiment.
    It categorizes the sentiment into Positive, Negative, Neutral, or Irrelevant.
""")

st.markdown("---")

# Input Section
user_input = st.text_area("✍️ Enter a tweet to analyze sentiment:", "", height=100)

if st.button("🚀 Analyze Sentiment"):
    if user_input.strip():
        cleaned_input = clean_text(user_input)
        transformed_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(transformed_input)[0]
        st.success(f"✅ **Predicted Sentiment:** {prediction}")
    else:
        st.warning("⚠️ Please enter a tweet to analyze.")

# Display Model Accuracy
st.markdown("---")
st.metric(label="📊 Model Accuracy on Validation Data", value=f"{validation_accuracy * 100:.2f}%")
