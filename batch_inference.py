import pandas as pd
import joblib

# Load the same trained model
model = joblib.load("model.pkl")

# Load unseen passenger records
unseen = pd.read_csv("unseen_passengers.csv")

# Select 20 passengers (rows 2 through 21)
batch = unseen.iloc[1:21].copy()

# Generate predictions for all 20 together
predictions = model.predict(batch)

# Add prediction results
batch["Predicted_Survived"] = predictions.astype(int)
batch["Prediction_Label"] = batch["Predicted_Survived"].map({
    1: "Survived",
    0: "Did not survive"
})

# Save results
batch.to_csv("batch_predictions.csv", index=False)

# Display results
print("BATCH INFERENCE RESULTS")
print("=======================")
print("Total passengers:", len(batch))
print(batch[["Predicted_Survived", "Prediction_Label"]].to_string(index=False))
print("\nSaved results to batch_predictions.csv")
