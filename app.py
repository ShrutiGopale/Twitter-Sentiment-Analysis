import streamlit as st
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
    
    return train, val

train_data, val_data = load_data()

# Text Preprocessing
def clean_text(text):
    if isinstance(text, str):  # Ensure only strings are processed
        text = text.lower()
        text = re.sub(r'[^A-Za-z0-9 ]+', ' ', text)
        return text
    return ""  # Return empty string for non-string values

train_data["clean_text"] = train_data["text"].apply(clean_text)
val_data["clean_text"] = val_data["text"].apply(clean_text)

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(
    train_data["clean_text"], train_data["type"], test_size=0.2, random_state=0
)

# Vectorization (Updated to avoid tokenizer warning)
vectorizer = CountVectorizer(stop_words=nltk.corpus.stopwords.words('english'))

X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Train Model
model = LogisticRegression(C=1, solver="liblinear", max_iter=200)
model.fit(X_train_bow, y_train)

# Streamlit UI
st.title("Twitter Sentiment Analysis")
st.write("Analyze the sentiment of tweets using a trained machine learning model.")

user_input = st.text_area("Enter a tweet to analyze sentiment:", "")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        cleaned_input = clean_text(user_input)
        transformed_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(transformed_input)[0]
        st.success(f"Predicted Sentiment: **{prediction}**")
    else:
        st.warning("Please enter a tweet to analyze.")

# Display Model Accuracy
st.write(f"Model Accuracy on Test Data: {accuracy_score(y_test, model.predict(X_test_bow)) * 100:.2f}%")

# Validation Accuracy
X_val_bow = vectorizer.transform(val_data["clean_text"])
y_val = val_data["type"]
val_accuracy = accuracy_score(y_val, model.predict(X_val_bow))
st.write(f"Model Accuracy on Validation Data: {val_accuracy * 100:.2f}%")
