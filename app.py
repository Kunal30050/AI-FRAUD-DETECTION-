from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load the model
model = joblib.load("model.pkl")  # make sure this path is correct

# Define input schema based on your dataset
class TransactionInput(BaseModel):
    transactionId: str
    amount: float
    location: str
    userId: str
    transactionDate: str
    paymentType: str
    userRiskScore: int
    deviceId: str
    ipAddress: str
    staticRuleStatus: str
    reviewRequired: str

@app.post("/predict")
async def predict(input: TransactionInput):
    # Convert input to DataFrame
    data = pd.DataFrame([input.dict()])
    
    # Predict using the model
    prediction = model.predict(data)[0]

    return {"prediction": int(prediction)}
