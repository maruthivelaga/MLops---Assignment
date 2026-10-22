import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict

# Create the FastAPI application
app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Machine learning model serving using FastAPI",
    version="1.0.0"
)

# Load the same model used for online, batch, and streaming inference
model = joblib.load("model.pkl")


# Validate incoming passenger data
class Passenger(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Pclass: int = Field(ge=1, le=3)
    Sex: str = Field(pattern="^(male|female)$")
    Age: float | None = Field(default=None, ge=0, le=100)
    SibSp: int = Field(ge=0, le=20)
    Parch: int = Field(ge=0, le=20)
    Fare: float = Field(ge=0)
    Embarked: str | None = Field(default=None, pattern="^[CSQ]$")


# Endpoint 1: Health check
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


# Endpoint 2: Predict passenger survival
@app.post("/predict")
def predict(passenger: Passenger):
    data = pd.DataFrame([passenger.model_dump()])

    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])

    return {
        "predicted_survived": prediction,
        "prediction_label": (
            "Survived" if prediction == 1 else "Did not survive"
        ),
        "survival_probability": round(probability, 4)
    }
