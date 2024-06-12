from sklearn.metrics import classification_report
import json

def load_predictions(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    true_labels = [item['type'] for item in data]
    predicted_labels = [item['predicted_type'] for item in data]
    return true_labels, predicted_labels

# Load the predictions
true_labels, predicted_labels = load_predictions('predictions.json')

# Calculate and print the classification report
report = classification_report(true_labels, predicted_labels, output_dict=False)
print(report)
