from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Load the AI/ML model and scaler
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Define input data format
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Prediction endpoint
@app.post("/predict")
def predict(data: DiabetesInput):
    # Convert input to NumPy array
    input_features = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure, 
                                data.SkinThickness, data.Insulin, data.BMI, 
                                data.DiabetesPedigreeFunction, data.Age]])

    # Normalize input using the saved scaler
    input_features = scaler.transform(input_features)

    # Make prediction
    prediction = model.predict(input_features)[0]

    # Return prediction as JSON
    return {"diabetes_prediction": int(prediction)}

