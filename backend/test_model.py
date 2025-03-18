import joblib

# Load the trained model & vectorizer
model = joblib.load("C:/PERSONAL/Coding/Projects/fake-news-detector/backend/model.pkl")
vectorizer = joblib.load("C:/PERSONAL/Coding/Projects/fake-news-detector/backend/vectorizer.pkl")

# Example news headlines to test
test_headlines = [
    "Breaking: NASA discovers alien life on Mars!",  # Likely Fake
    "Government announces new tax policies for 2025",  # Likely Real
    "You won't believe this shocking celebrity scandal!",  # Likely Fake
    "Stock markets rise as inflation slows down",  # Likely Real
]

# Convert text to TF-IDF vectors
X_test = vectorizer.transform(test_headlines)

# Get model predictions
predictions = model.predict(X_test)

# Print results
for i, headline in enumerate(test_headlines):
    print(f"🔹 {headline} → Prediction: {'FAKE' if predictions[i] == 1 else 'REAL'}")
