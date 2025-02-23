from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Load the scaler and the trained random forest model
scaler = joblib.load("scaler.pkl")
model = joblib.load("rf_model.pkl")

# Define the input data format
class DiabetesInput(BaseModel):
    Glucose: float
    BMI: float
    Age: int
    BloodPressure: float
    DiabetesPedigreeFunction: float

# Prediction endpoint
@app.post("/predict")
def predict(data: DiabetesInput):
    # Create input array
    input_features = np.array([[data.Glucose, data.BMI, data.Age, data.BloodPressure, data.DiabetesPedigreeFunction]])
    # Apply the same scaling transformation as during training
    input_scaled = scaler.transform(input_features)
    prediction = model.predict(input_scaled)[0]
    return {"diabetes_prediction": int(prediction)}
