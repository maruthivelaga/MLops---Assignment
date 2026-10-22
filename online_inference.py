import pandas as pd
import joblib

# Load the previously trained model
model = joblib.load("model.pkl")

# Load the unseen passenger records
unseen = pd.read_csv("unseen_passengers.csv")

# Select one passenger
passenger = unseen.iloc[[0]]

print("Passenger details:")
print(passenger.to_string(index=False))

# Predict survival
prediction = int(model.predict(passenger)[0])
probability = float(model.predict_proba(passenger)[0][1])

print("\nOnline Inference Result")
print("-----------------------")
print("Predicted survival:", prediction)
print(
    "Prediction:",
    "Survived" if prediction == 1 else "Did not survive"
)
print(f"Survival probability: {probability:.2%}")
