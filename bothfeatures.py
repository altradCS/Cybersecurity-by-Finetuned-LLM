import streamlit as st
from transformers import pipeline, DistilBertTokenizer, DistilBertTokenizer, DistilBertForSequenceClassification, AdamW, AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from sklearn.metrics import accuracy_score

# Load pre-trained sentiment analysis model
nlptown = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(nlptown)
model_nlptown = AutoModelForSequenceClassification.from_pretrained(nlptown)
sentiment_classifier = pipeline("sentiment-analysis", model=model_nlptown , tokenizer=tokenizer)

# Load pre-trained DistilBERT model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer_sentiment = DistilBertTokenizer.from_pretrained(model_name)
model_sentiment = DistilBertForSequenceClassification.from_pretrained(model_name)


def analyze_text(text1):
    # Tokenize and process the input text
    inputs = tokenizer_sentiment(text1, return_tensors="pt", truncation=True)

    # Get the model's prediction
    with torch.no_grad():
        outputs = model_sentiment(**inputs)

    # Interpret the model's output for fraud detection
    prediction = torch.argmax(outputs.logits, dim=1).item()

    return prediction


def display_result(result1):
    st.subheader("Analysis Result:")

    if result1 == 1:
        st.error("Fraudulent activity detected!")
    else:
        st.success("No fraudulent activity detected.")


def analyze_text_sentiment(text2):
    classifier = pipeline("sentiment-analysis", model=model_sentiment, tokenizer=tokenizer_sentiment)
    result = classifier(text2)
    return result


def display_sentiment_result(result2):
    st.subheader("Sentiment Analysis Result:")
    st.write(f"Sentiment: {result2[0]['label']} with confidence {result2[0]['score']:.4f}")


# Streamlit app
def main():
    st.title("Sentiment Cybersecurity Forensics Webapp")
    # User input for text
    text1 = st.text_area("Enter your text for analysis:", "")

    if st.button("Analyze with LLM-1"):
        if text1:
            # analyze the text for fraudulent activities
            fraud_detection_result = analyze_text(text1)
            display_result(fraud_detection_result)
        else:
            st.warning("Please enter text for analysis.")

    # Text analysis for cybersecurity forensics
    # st.subheader("Text Analysis for Cybersecurity Forensics")
    text2 = st.text_area("Enter text for analysis :", "")
    if st.button("Analyze with LLM-2"):
        if text2:
            sentiment_result = analyze_text_sentiment(text2)
            display_sentiment_result(sentiment_result)
        else:
            st.warning("Please enter text for analysis.")


    # 1. Suspicious Text Analysis
    st.subheader("1. Suspicious Text Analysis")
    text_suspicious = st.text_area("Enter text for suspicious analysis:", "")
    if st.button("Analyze Suspicious Text"):
        if text_suspicious:
            result_suspicious = sentiment_classifier(text_suspicious)
            display_sentiment_result("Suspicious Text Analysis Result", result_suspicious)
        else:
            st.warning("Please enter text for analysis.")

    # 2. Fraudulent, Spam, Harm, Viruses, Malware, Ransomware Analysis
    st.subheader("2. Fraudulent, Spam, Harm, Viruses, Malware, Ransomware Analysis")
    text_cybersecurity = st.text_area("Enter text for cybersecurity analysis:", "")
    if st.button("Analyze Cybersecurity Text"):
        if text_cybersecurity:
            result_cybersecurity = sentiment_classifier(text_cybersecurity)
            display_sentiment_result("Cybersecurity Text Analysis Result", result_cybersecurity)
        else:
            st.warning("Please enter text for analysis.")


    # 5. Option to test normal and suspicious content after fine-tuning
    st.subheader("5. Test Normal and Suspicious Content after Fine-tuning")
    text_test = st.text_area("Enter text for testing:", "")
    if st.button("Test Text After Fine-tuning"):
        if text_test and 'df_finetune' in locals():
            result_test = sentiment_classifier(text_test)
            display_sentiment_result("Test Text Analysis Result", result_test)
        else:
            st.warning("Please fine-tune the model first and enter text for testing.")


if __name__ == "__main__":
    main()
