from fastapi import FastAPI, Request
import pickle
import numpy as np

app = FastAPI()

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)
    return {"prediction": prediction.tolist()}
