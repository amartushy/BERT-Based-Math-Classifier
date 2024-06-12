import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def load_data(file_path):
    try:
        # Load data
        data = pd.read_json(file_path)
        print("Data loaded successfully.")
        return data
    except Exception as e:
        print(f"Failed to load data: {e}")
        return None

def prepare_features(vectorizer, data, text_column='problem'):
    try:
        # Transform the problem descriptions using the provided vectorizer
        X = vectorizer.transform(data[text_column])
        return X
    except Exception as e:
        print(f"Failed to prepare features: {e}")

def main():
    train_data = load_data('combined_train.json')
    test_data = load_data('combined_test.json')

    if train_data is not None and test_data is not None:
        # Initialize TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', max_features=5000)
        vectorizer.fit(train_data['problem'])  # Fit only on training data

        # Prepare features
        X_train = prepare_features(vectorizer, train_data)
        X_test = prepare_features(vectorizer, test_data)

        # Prepare the target variables
        y_train = train_data['type']
        y_test = test_data['type']

        # Initialize and train the Logistic Regression model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        print("Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()

