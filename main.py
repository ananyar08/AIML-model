from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model = joblib.load("diabetes_model.pkl")

# Define input data format
class DiabetesInput(BaseModel):
    Glucose: float
    BMI: float
    Age: int
    BloodPressure: float
    DiabetesPedigreeFunction: float

# Prediction endpoint
@app.post("/predict")
def predict(data: DiabetesInput):
    # Convert input to NumPy array
    input_features = np.array([[data.Glucose, data.BMI, data.Age, data.BloodPressure, data.DiabetesPedigreeFunction]])
    
    # Make prediction
    prediction = model.predict(input_features)[0]

    return {"diabetes_prediction": int(prediction)}


