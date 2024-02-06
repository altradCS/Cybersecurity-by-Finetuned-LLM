
import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load pre-trained DistilBERT model and tokenizer
model_name = 'D:/Cybersecurity/streamlitapps/DistBert'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

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

if __name__ == "__main__":
    main()
