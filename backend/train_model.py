import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load fake news dataset & add label 1
df_fake = pd.read_csv("C:/PERSONAL/Coding/Projects/fake-news-detector/dataset/fake_news.csv")
df_fake["label"] = 1  # Fake news = 1

# Load real news dataset & add label 0
df_real = pd.read_csv("C:/PERSONAL/Coding/Projects/fake-news-detector/dataset/real_news.csv")
df_real["label"] = 0  # Real news = 0

# Merge both datasets
df = pd.concat([df_fake, df_real], ignore_index=True)

# Keep only title & label
df = df[["title", "label"]]  

# Text processing
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(df["title"])
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model & vectorizer
joblib.dump(model, "C:/PERSONAL/Coding/Projects/fake-news-detector/backend/model.pkl")
joblib.dump(vectorizer, "C:/PERSONAL/Coding/Projects/fake-news-detector/backend/vectorizer.pkl")

print("✅ Model training complete! Saved as model.pkl.")
