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

# Load pre-trained DistilBERT model and tokenizer for fine-tuning
tokenizer_finetune = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model_finetune = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Function to fine-tune DistilBERT model with a custom dataset
def fine_tune_distilbert(dataset, num_epochs=3):
    optimizer = AdamW(model_finetune.parameters(), lr=5e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Tokenize and preprocess the dataset
    inputs = tokenizer_finetune(dataset['prompt'].tolist(), return_tensors="pt", padding=True, truncation=True)
    labels = torch.tensor(dataset['label'].tolist())

    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoader for training and validation sets
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)


        # Validation
        model_finetune.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids, attention_mask, labels = batch
                outputs = model_finetune(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                logits = outputs.logits
                _, predictions = torch.max(logits, dim=1)
                correct_predictions += torch.sum(predictions == labels).item()
                total_samples += len(labels)

        avg_val_loss = val_loss / len(val_dataloader)
        accuracy = correct_predictions / total_samples

        print(f"Epoch {epoch + 1}/{num_epochs}: Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")

    return model_finetune

# Streamlit app
def main():
    st.title("Cybersecurity Forensics and Fine-tuning App")

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
