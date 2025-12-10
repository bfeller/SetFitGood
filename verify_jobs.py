import requests
import time
import sys

BASE_URL = "http://localhost:8000"

def wait_for_health():
    print("Waiting for API to be ready...")
    for _ in range(60): # Increased wait time for heavier image
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

def test_async_train():
    print("Testing async training endpoint...")
    payload = {
        "model_name": "async_test_model",
        "base_model": "paraphrase-MiniLM-L3-v2",
        "examples": [
            {"text": "I love this product", "label": "positive"},
            {"text": "This is terrible", "label": "negative"},
            {"text": "It is okay", "label": "neutral"}
        ],
        "num_iterations": 2,
        "batch_size": 2
    }
    resp = requests.post(f"{BASE_URL}/v1/train", json=payload)
    if resp.status_code == 200:
        job = resp.json()
        print("Training job submitted:", job)
        return job["job_id"]
    else:
        print(f"Training failed: {resp.text}")
        sys.exit(1)

def poll_job(job_id):
    print(f"Polling job {job_id}...")
    for _ in range(30):
        resp = requests.get(f"{BASE_URL}/v1/jobs/{job_id}")
        if resp.status_code == 200:
            status = resp.json()
            print(f"Job status: {status['status']}")
            if status['status'] == 'completed':
                print("Job completed!")
                return
            if status['status'] == 'failed':
                print(f"Job failed: {status}")
                sys.exit(1)
        else:
            print(f"Error checking job: {resp.text}")
        time.sleep(2)
    print("Timeout waiting for job completion")
    sys.exit(1)

def check_metadata(model_name):
    print(f"Checking metadata for {model_name}...")
    resp = requests.get(f"{BASE_URL}/v1/models/{model_name}")
    if resp.status_code == 200:
        meta = resp.json()
        print("Metadata:", meta)
        if "accuracy" in meta:
             print("Metadata verified.")
    else:
        print(f"Failed to get metadata: {resp.text}")
        sys.exit(1)

def check_embeddings(model_name):
    print(f"Checking embeddings for {model_name}...")
    payload = {
        "model_name": model_name,
        "texts": ["This is a test sentence for embeddings."]
    }
    resp = requests.post(f"{BASE_URL}/v1/embeddings", json=payload)
    if resp.status_code == 200:
        data = resp.json()
        print("Embeddings received, shape:", len(data["embeddings"]), "x", len(data["embeddings"][0]))
    else:
        print(f"Failed to get embeddings: {resp.text}")
        sys.exit(1)

if __name__ == "__main__":
    wait_for_health()
    job_id = test_async_train()
    poll_job(job_id)
    check_metadata("async_test_model")
    check_embeddings("async_test_model")
