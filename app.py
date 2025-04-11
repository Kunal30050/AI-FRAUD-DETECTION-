from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Define the structure of incoming transaction data
class Transaction(BaseModel):
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

@app.get("/")
def read_root():
    return {"message": "Fraud Detection API is up and running!"}

@app.post("/predict")
async def predict(transaction: Transaction):
    # Convert the received transaction to a dictionary
    data = transaction.dict()

    # TODO: Load and use your ML model to make a prediction here
    # For now, return the received data and dummy prediction
    return {
        "prediction": "Not Fraud",  # Replace this with actual model output
        "received_data": data
    }
