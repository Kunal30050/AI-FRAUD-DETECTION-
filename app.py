from fastapi import FastAPI, Request
import pickle
import pandas as pd

app = FastAPI()

# Load your trained model
model = pickle.load(open("model.pkl", "rb"))

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()

    # Convert the input JSON to DataFrame
    df = pd.DataFrame([data])

    # Get prediction
    prediction = model.predict(df)[0]

    return {
        "prediction": int(prediction)
    }
