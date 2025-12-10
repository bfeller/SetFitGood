import requests
import time
import sys

BASE_URL = "http://localhost:8000"

def wait_for_health():
    print("Waiting for API to be ready...")
    for _ in range(30):
        try:
            resp = requests.get(f"{BASE_URL}/health")
            if resp.status_code == 200:
                print("API is ready!")
                return
        except requests.ConnectionError:
            pass
        time.sleep(2)
    print("API failed to start.")
    sys.exit(1)

def test_train():
    print("Testing training endpoint...")
    payload = {
        "model_name": "test_model_v1",
        "base_model": "paraphrase-MiniLM-L3-v2", # Smaller model for faster test
        "examples": [
            {"text": "I love this product", "label": "positive"},
            {"text": "This is the best thing ever", "label": "positive"},
            {"text": "I hate this", "label": "negative"},
            {"text": "This is terrible", "label": "negative"},
            {"text": "The service was okay", "label": "neutral"},
            {"text": "It is acceptable", "label": "neutral"}
        ],
        "num_iterations": 2, # minimal for test
        "batch_size": 2
    }
    resp = requests.post(f"{BASE_URL}/v1/train", json=payload)
    if resp.status_code == 200:
        print("Training started successfully.")
    else:
        print(f"Training failed: {resp.text}")
        sys.exit(1)

def test_predict():
    print("Waiting for training to potentially finish (sleeping 30s)...")
    time.sleep(30) # Wait for background training to finish
    
    print("Testing prediction endpoint...")
    payload = {
        "model_name": "test_model_v1",
        "texts": ["This is amazing", "This is bad"]
    }
    resp = requests.post(f"{BASE_URL}/v1/predict", json=payload)
    if resp.status_code == 200:
        data = resp.json()
        print("Prediction response:", data)
        # Verify predictions
        preds = data["predictions"]
        if preds[0]["label"] == "positive" and preds[1]["label"] == "negative":
             print("Predictions look correct!")
        else:
             print("Predictions might be inaccurate (expected given tiny training set), but endpoint works.")
    else:
        print(f"Prediction failed: {resp.text}")
        sys.exit(1)

if __name__ == "__main__":
    wait_for_health()
    test_train()
    test_predict()
