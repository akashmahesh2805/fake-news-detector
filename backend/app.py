from fastapi import FastAPI
import joblib

# Load the trained model & vectorizer
model = joblib.load("C:/PERSONAL/Coding/Projects/fake-news-detector/backend/model.pkl")
vectorizer = joblib.load("C:/PERSONAL/Coding/Projects/fake-news-detector/backend/vectorizer.pkl")

app = FastAPI()

@app.post("/predict")
def predict(data: dict):
    text = data["title"]
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    return {"prediction": "FAKE" if prediction == 1 else "REAL"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
