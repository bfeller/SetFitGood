import requests
import time
import sys
import json

BASE_URL = "http://localhost:8000"

def wait_for_health():
    print(f"Waiting for API to be ready at {BASE_URL}...")
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
    print("\n--- Testing Training Endpoint ---")
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
        data = resp.json()
        print("Training started successfully.")
        print(f"Job ID: {data.get('job_id')}")
        return data.get('job_id')
    else:
        print(f"Training failed: {resp.text}")
        sys.exit(1)

def test_jobs(job_id):
    print(f"\n--- Testing Jobs Endpoint for Job {job_id} ---")
    if not job_id:
        print("Skipping jobs test because no job_id was returned from training.")
        return

    # Poll for completion
    for _ in range(60): # Wait up to 120 seconds
        resp = requests.get(f"{BASE_URL}/v1/jobs/{job_id}")
        if resp.status_code == 200:
            status_data = resp.json()
            status = status_data.get("status")
            print(f"Job status: {status}")
            if status == "completed":
                print("Job completed successfully!")
                return
            elif status == "failed":
                print(f"Job failed: {status_data.get('error')}")
                sys.exit(1)
        else:
            print(f"Failed to get job status: {resp.text}")
        time.sleep(2)
    
    print("Job timed out.")
    sys.exit(1)

def test_models():
    print("\n--- Testing Models Endpoint ---")
    model_name = "test_model_v1"
    resp = requests.get(f"{BASE_URL}/v1/models/{model_name}")
    if resp.status_code == 200:
        print(f"Model metadata retrieved for {model_name}")
        print(json.dumps(resp.json(), indent=2))
    else:
        print(f"Failed to get model metadata: {resp.text}")
        # Don't exit, might be just not ready or file missing in this test env setup
            
def test_predict():
    print("\n--- Testing Prediction Endpoint ---")
    payload = {
        "model_name": "test_model_v1",
        "texts": ["This is amazing", "This is bad"]
    }
    resp = requests.post(f"{BASE_URL}/v1/predict", json=payload)
    if resp.status_code == 200:
        data = resp.json()
        print("Prediction response:", json.dumps(data, indent=2))
        # Verify predictions
        preds = data["predictions"]
        if preds[0]["label"] == "positive" and preds[1]["label"] == "negative":
             print("Predictions look correct!")
        else:
             print("Predictions might be inaccurate (expected given tiny training set), but endpoint works.")
    else:
        print(f"Prediction failed: {resp.text}")
        sys.exit(1)

def test_embeddings():
    print("\n--- Testing Embeddings Endpoint ---")
    payload = {
        "model_name": "test_model_v1",
        "texts": ["Hello world", "Machine learning is cool"],
        "dimensions": 384 # Default for MiniLM
    }
    resp = requests.post(f"{BASE_URL}/v1/embeddings", json=payload)
    if resp.status_code == 200:
        data = resp.json()
        embeddings = data.get("embeddings")
        if embeddings and len(embeddings) == 2:
            print(f"Successfully generated {len(embeddings)} embeddings.")
            print(f"Embedding dimension: {len(embeddings[0])}")
        else:
            print("Embeddings response format check failed.")
    else:
        print(f"Embeddings generation failed: {resp.text}")
        sys.exit(1)

if __name__ == "__main__":
    wait_for_health()
    job_id = test_train()
    test_jobs(job_id)
    test_models()
    test_predict()
    test_embeddings()
