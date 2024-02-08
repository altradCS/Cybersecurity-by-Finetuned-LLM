import streamlit as st
from transformers import DistilBertTokenizer, pipeline, DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
import torch

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

    if result == 1:
        st.error("Fraudulent activity detected!")
    else:
        st.success("No fraudulent activity detected.")


def analyze_text_sentiment(text2):
    classifier = pipeline("sentiment-analysis", model=model_sentiment, tokenizer=tokenizer_sentiment)
    result = classifier(text)
    return result


def display_sentiment_result(result2):
    st.subheader("Sentiment Analysis Result:")
    st.write(f"Sentiment: {result[0]['label']} with confidence {result[0]['score']:.4f}")


# Streamlit app
def main():
    st.title("Fraud Detection Web App")
    # User input for text
    text1 = st.text_area("Enter your text for analysis:", "")

    if st.button("Analyze"):
        if text1:
            #analyze the text for fraudulent activities
            fraud_detection_result = analyze_text(text1)
            display_result(fraud_detection_result)
        else:
            st.warning("Please enter text for analysis.")

    st.write("Cybersecurity Forensics Webapp")
    # Text analysis for cybersecurity forensics
    #st.subheader("Text Analysis for Cybersecurity Forensics")
    text2 = st.text_area("Enter text for analysis :", "")
    if st.button("Evaluate"):
        if text2:
            sentiment_result = analyze_text_sentiment(text2)
            display_sentiment_result(sentiment_result)
        else:
            st.warning("Please enter text for analysis.")


if __name__ == "__main__":
    main()
