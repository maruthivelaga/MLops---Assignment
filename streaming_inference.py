import time
import pandas as pd
import joblib

# Load the same saved model
model = joblib.load("model.pkl")

# Load unseen passenger records
unseen = pd.read_csv("unseen_passengers.csv")

# Select 10 passengers (rows 22 through 31)
stream = unseen.iloc[21:31]

if len(stream) != 10:
    raise ValueError(
        "Expected 10 streaming records. Run train_model.py first."
    )

print("STREAMING INFERENCE STARTED")
print("============================")

# Process one passenger at a time
for index, (_, passenger) in enumerate(
    stream.iterrows(), start=1
):
    record = passenger.to_frame().T

    prediction = int(model.predict(record)[0])
    probability = float(model.predict_proba(record)[0][1])

    print(f"\nIncoming record: {index}/10")
    print("Predicted survival:", prediction)
    print(
        "Prediction:",
        "Survived" if prediction == 1 else "Did not survive"
    )
    print(f"Survival probability: {probability:.2%}")

    # Wait one second before processing the next record
    if index < len(stream):
        time.sleep(1)

print("\nStreaming inference completed.")
