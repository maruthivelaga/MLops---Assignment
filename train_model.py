import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
DATA_PATH = "data/train.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "Dataset not found. Place Kaggle train.csv in the data/ folder."
    )

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

# 2. Separate features and target
X = df.drop(
    columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"]
)
y = df["Survived"]

# 3. Define feature columns
numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Sex", "Embarked"]

# 4. Preprocess numeric data
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 5. Preprocess categorical data
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 6. Create complete ML pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

# 7. Split labelled data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 8. Train the model once
model.fit(X_train, y_train)

# 9. Evaluate on held-out test data
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest accuracy: {accuracy:.4f}")
print(f"Test accuracy percentage: {accuracy * 100:.2f}%")
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# 10. Save the complete trained pipeline
joblib.dump(model, "model.pkl")
print("\nSaved trained pipeline: model.pkl")

# 11. Reserve 31 held-out records for inference demonstrations
# Record 1: online; records 2-21: batch; records 22-31: streaming.
unseen = X_test.iloc[:31].copy()
unseen.to_csv("unseen_passengers.csv", index=False)

print("Saved unseen records:", len(unseen))
print("Saved file: unseen_passengers.csv")
