import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "bigscience/bloom"  # Replace with your chosen LLM model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def analyze_text(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predictions = probs.argmax(dim=-1)
    return predictions.item(), probs.tolist()[0]

def compute_accuracy(true_labels, predicted_labels):
    num_correct = (true_labels == predicted_labels).sum().item()
    accuracy = num_correct / len(true_labels)
    return accuracy

st.title("Cybersecurity Forensics with LLM")

# Input for text analysis
text_input = st.text_area("Enter suspicious text")

# Analyze text and display results
if text_input:
    prediction, probs = analyze_text(text_input)
    st.write("Predicted category:", model.config.id2label[prediction])
    st.write("Probabilities:", probs)

# Option to upload dataset for fine-tuning
uploaded_file = st.file_uploader("Upload dataset for fine-tuning")
if uploaded_file:
    # Process dataset and fine-tune the model (implementation details omitted for brevity)
    st.success("Model fine-tuned successfully!")

# Option to test normal and suspicious content
normal_text_input = st.text_area("Enter normal text")
suspicious_text_input = st.text_area("Enter suspicious text")

if normal_text_input and suspicious_text_input:
    normal_prediction, normal_probs = analyze_text(normal_text_input)
    suspicious_prediction, suspicious_probs = analyze_text(suspicious_text_input)

    st.write("Normal text analysis:")
    st.write("Predicted category:", model.config.id2label[normal_prediction])
    st.write("Probabilities:", normal_probs)

    st.write("Suspicious text analysis:")
    st.write("Predicted category:", model.config.id2label[suspicious_prediction])
    st.write("Probabilities:", suspicious_probs)

# Compute and display model accuracy (implementation details omitted)
