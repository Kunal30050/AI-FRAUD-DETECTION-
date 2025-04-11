from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import os

app = FastAPI()

# Allow CORS (for testing or frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(model_path)

@app.get("/")
def read_root():
    return {"message": "AI Fraud Detection is Live 🎉"}

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()

    # Extract relevant features for the model
    features = [
        data["amount"],
        data["userRiskScore"],
        # Add more fields in the order expected by your model
    ]

    # Convert to array and reshape
    input_data = np.array(features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data)[0]

    label = "Fraud" if prediction == 1 else "Not Fraud"

    return {
        "prediction": label,
        "received_data": data
    }
