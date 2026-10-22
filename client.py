import requests

BASE_URL = "http://127.0.0.1:8000"

# 1. Test the health endpoint
try:
    health_response = requests.get(
        f"{BASE_URL}/health",
        timeout=10
    )

    print("HEALTH CHECK")
    print("Status code:", health_response.status_code)
    print("Response:", health_response.json())

    # 2. Prepare passenger details
    passenger = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }

    # 3. Send passenger details to the prediction endpoint
    response = requests.post(
        f"{BASE_URL}/predict",
        json=passenger,
        timeout=10
    )

    print("\nPREDICTION RESULT")
    print("Status code:", response.status_code)
    print("Response:", response.json())

    response.raise_for_status()

except requests.RequestException as error:
    print("API request failed:", error)
    print("Ensure the FastAPI server is running.")

