# Twitter-Sentiment-Analysis

Overview

This project is a Twitter Sentiment Analysis web application built using Streamlit and Scikit-learn. The application takes a tweet as input and predicts its sentiment as Positive and Negative using a trained Logistic Regression model.

Features

Sentiment Classification: Predicts the sentiment of a given tweet.
Machine Learning Model: Uses a Bag-of-Words (BoW) model with Logistic Regression.
Performance Metrics: Displays the model's accuracy on validation data.
User-Friendly Interface: Styled with a modern design using Streamlit.

Dataset

The model is trained on a dataset consisting of:
twitter_training.csv (Training Data)
twitter_validation.csv (Validation Data)

Model Training

The sentiment classification model is trained using:
Text Preprocessing: Removing special characters, lowercasing, tokenization.
Vectorization: Bag-of-Words (BoW) model using CountVectorizer.
Classifier: Logistic Regression with liblinear solver.
