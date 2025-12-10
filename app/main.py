from fastapi import FastAPI, HTTPException, BackgroundTasks, Security, Depends, status
from fastapi.security import APIKeyHeader
from .schemas import TrainRequest, PredictRequest, PredictResponse, PredictionResult, JobStatus, ModelMetadata, EmbeddingRequest, EmbeddingResponse
from .model_manager import ModelManager
import logging
import os
import uuid
import time
import json
from datetime import datetime
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SetFit API", version="0.1.0")
model_manager = ModelManager()

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# In-memory job store (replace with Redis/DB for persistence if needed)
jobs: Dict[str, JobStatus] = {}

async def get_api_key(api_key_header: str = Security(api_key_header)):
    expected_api_key = os.getenv("API_KEY")
    if expected_api_key:
        if api_key_header == expected_api_key:
            return api_key_header
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )
    return None

def run_training_job(job_id: str, request: TrainRequest):
    try:
        jobs[job_id].status = "running"
        model_manager.train_model(
            model_name=request.model_name,
            base_model_name=request.base_model,
            examples=request.examples,
            num_iterations=request.num_iterations,
            learning_rate=request.learning_rate,
            batch_size=request.batch_size
        )
        jobs[job_id].status = "completed"
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        jobs[job_id].status = "failed"
        jobs[job_id].error = str(e)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/v1/train", dependencies=[Depends(get_api_key)])
def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    job = JobStatus(
        job_id=job_id, 
        status="pending", 
        model_name=request.model_name,
        created_at=time.time()
    )
    jobs[job_id] = job
    
    background_tasks.add_task(run_training_job, job_id, request)
    return job

@app.get("/v1/jobs/{job_id}", dependencies=[Depends(get_api_key)])
def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@app.get("/v1/models/{model_name}", dependencies=[Depends(get_api_key)])
def get_model_metadata(model_name: str):
    try:
        # Check if metadata exists
        model_path = model_manager._get_model_path(model_name)
        metadata_path = os.path.join(model_path, "metadata.json")
        if not os.path.exists(metadata_path):
             raise HTTPException(status_code=404, detail="Model metadata not found (model might exist but have no metadata)")
        
        with open(metadata_path, "r") as f:
            data = json.load(f)
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/predict", response_model=PredictResponse, dependencies=[Depends(get_api_key)])
def predict(request: PredictRequest):
    try:
        predictions = model_manager.predict(request.model_name, request.texts)
        results = []
        for i, text in enumerate(request.texts):
            results.append(PredictionResult(text=text, label=predictions[i]))
        return PredictResponse(predictions=results)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/embeddings", response_model=EmbeddingResponse, dependencies=[Depends(get_api_key)])
def get_embeddings(request: EmbeddingRequest):
    try:
        embeddings = model_manager.get_embeddings(request.model_name, request.texts)
        return EmbeddingResponse(embeddings=embeddings)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error during embedding generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
