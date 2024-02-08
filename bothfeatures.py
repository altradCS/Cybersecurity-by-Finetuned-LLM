import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification, AdamW

# Load pre-trained DistilBERT model and tokenizer
model_name1 = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = DistilBertTokenizer.from_pretrained(model_name1)
model = DistilBertForSequenceClassification.from_pretrained(model_name1)

def analyze_text(text):
    # Tokenize and process the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    # Get the model's prediction
    with torch.no_grad():
        outputs = model(**inputs)

    # Interpret the model's output for fraud detection
    prediction = torch.argmax(outputs.logits, dim=1).item()

    return prediction


def display_result(result):
    st.subheader("Analysis Result:")

    if result == 1:
        st.error("Fraudulent activity detected!")
    else:
        st.success("No fraudulent activity detected.")



################################


# Load pre-trained DistilBERT model and tokenizer for sentiment analysis
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer_sentiment = DistilBertTokenizer.from_pretrained(model_name)
model_sentiment = DistilBertForSequenceClassification.from_pretrained(model_name)
  

def analyze_text_sentiment(text):
    classifier = pipeline("sentiment-analysis", model=model_sentiment, tokenizer=tokenizer_sentiment)
    result = classifier(text)
    return result

def display_sentiment_result(result):
    st.subheader("Sentiment Analysis Result:")
    st.write(f"Sentiment: {result[0]['label']} with confidence {result[0]['score']:.4f}")


# Streamlit app
def main():
    st.title("Fraud Detection Web App")

    # User input for text
    text = st.text_area("Enter text for analysis:", "")

    if st.button("Analyze"):
        if text:
            # Analyze the text for fraudulent activities
            fraud_detection_result = analyze_text(text)
            display_result(fraud_detection_result)
        else:
            st.warning("Please enter text for analysis.")

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

if __name__ == "__main__":
    main()
