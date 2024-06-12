import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
import logging

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()  # Set the model to evaluation mode
    logger.info("Model and tokenizer loaded for inference")
    return model, tokenizer

def predict(model, tokenizer, texts, batch_size=10):
    results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            results.extend(predictions.argmax(dim=-1).tolist())
        logger.info(f"Processed batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")
    return results

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    problems = [item['problem'] for item in data]
    labels = [item['type'] for item in data]
    return problems, labels, data

# Define the class labels expected by the model
class_labels = ['Algebra', 'Counting & Probability', 'Geometry', 'Intermediate Algebra', 'Number Theory', 'Prealgebra', 'Precalculus']

# Load the model
model_path = './model'
model, tokenizer = load_model(model_path)

# Load test data
test_data_path = 'combined_test.json'
problems, labels, full_data = load_data(test_data_path)

# Make predictions
predictions = predict(model, tokenizer, problems)
predicted_labels = [class_labels[idx] for idx in predictions]

# Save the predictions alongside the actual label and problem to JSON
for item, prediction in zip(full_data, predicted_labels):
    item['predicted_type'] = prediction

# Save the augmented data with predictions to a JSON file
with open('predictions.json', 'w') as f:
    json.dump(full_data, f, indent=4)

logger.info("Predictions saved to predictions.json")

