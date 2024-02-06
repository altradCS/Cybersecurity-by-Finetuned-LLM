import streamlit as st
from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
import pandas as pd
from tqdm import tqdm

# Load pre-trained DistilBERT model and tokenizer for sentiment analysis
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer_sentiment = DistilBertTokenizer.from_pretrained(model_name)
model_sentiment = DistilBertForSequenceClassification.from_pretrained(model_name)

# Streamlit app
def main():
    st.title("Cybersecurity Forensics Webapp")

    # Text analysis for cybersecurity forensics
    st.subheader("Text Analysis for Cybersecurity Forensics")
    text = st.text_area("Enter text for analysis:", "")
    if st.button("Analyze"):
        if text:
            sentiment_result = analyze_text_sentiment(text)
            display_sentiment_result(sentiment_result)
        else:
            st.warning("Please enter text for analysis.")

def analyze_text_sentiment(text):
    classifier = pipeline("sentiment-analysis", model=model_sentiment, tokenizer=tokenizer_sentiment)
    result = classifier(text)
    return result

def display_sentiment_result(result):
    st.subheader("Sentiment Analysis Result:")
    st.write(f"Sentiment: {result[0]['label']} with confidence {result[0]['score']:.4f}")


if __name__ == "__main__":
    main()
