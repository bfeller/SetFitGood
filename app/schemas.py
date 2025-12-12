from pydantic import BaseModel
from typing import List, Optional

class TrainingExample(BaseModel):
    text: str
    label: str

class TrainRequest(BaseModel):
    model_name: str = "default_model"
    base_model: str = "Alibaba-NLP/gte-modernbert-base"
    examples: List[TrainingExample]
    num_iterations: int = 20
    learning_rate: float = 2e-5
    batch_size: int = 16

class PredictRequest(BaseModel):
    model_name: str = "default_model"
    texts: List[str]

class EmbeddingRequest(BaseModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    texts: List[str]

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]

class PredictionResult(BaseModel):
    text: str
    label: str
    score: Optional[float] = None

class PredictResponse(BaseModel):
    predictions: List[PredictionResult]

class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, running, completed, failed
    model_name: str
    created_at: float
    error: Optional[str] = None

class ModelMetadata(BaseModel):
    model_name: str
    base_model: str
    created_at: str
    accuracy: Optional[float] = None
    training_duration_seconds: Optional[float] = None
    num_examples: int
