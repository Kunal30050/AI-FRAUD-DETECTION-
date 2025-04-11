from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
class PredictionRequest(BaseModel):
    features: list

model = joblib.load("model.pkl")  # or "random_forest_model.pkl"

@app.post("/predict")
def predict(data: PredictionRequest):
    features = [data.features]  # Make it 2D for scikit-learn
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}
