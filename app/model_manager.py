import os
from typing import List, Dict
from setfit import SetFitModel, SetFitTrainer, Trainer, TrainingArguments
from datasets import Dataset
import logging
from .schemas import TrainingExample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = os.getenv("MODEL_PATH", "/app/models")

class ModelManager:
    def __init__(self):
        self.loaded_models: Dict[str, SetFitModel] = {}
        # Ensure models directory exists
        os.makedirs(MODELS_DIR, exist_ok=True)

    def _get_model_path(self, model_name: str) -> str:
        return os.path.join(MODELS_DIR, model_name)

    def load_model(self, model_name: str) -> SetFitModel:
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        # Check for Flash Attention availability
        import torch
        use_flash_attn = False
        try:
            import flash_attn
            if torch.cuda.is_available():
                use_flash_attn = True
                logger.info("Flash Attention 2 is available and will be used.")
            else:
                 logger.info("Flash Attention library found but CUDA is not available.")
        except ImportError:
            logger.info("Flash Attention library not found.")

        model_kwargs = {}
        if use_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            model_kwargs["torch_dtype"] = torch.float16 # FA2 requires fp16 or bf16

        model_path = self._get_model_path(model_name)
        if not os.path.exists(model_path):
             # check if it is a huggungface model
            try:
                logger.info(f"Loading model {model_name} from HuggingFace Hub or local path with kwargs: {model_kwargs}...")
                model = SetFitModel.from_pretrained(model_name, model_kwargs=model_kwargs, trust_remote_code=True)
                self.loaded_models[model_name] = model
                return model
            except Exception as e:
                logger.error(f"Model {model_name} not found locally or on HF Hub: {e}")
                raise ValueError(f"Model {model_name} not found.")
        
        logger.info(f"Loading model {model_name} from {model_path} with kwargs: {model_kwargs}...")
        model = SetFitModel.from_pretrained(model_path, model_kwargs=model_kwargs, trust_remote_code=True)
        self.loaded_models[model_name] = model
        return model

    def train_model(self, 
                    model_name: str, 
                    base_model_name: str, 
                    examples: List[TrainingExample],
                    num_iterations: int = 20,
                    learning_rate: float = 2e-5,
                    batch_size: int = 16):
        
        logger.info(f"Starting training for model: {model_name} using base: {base_model_name}")
        
        # Custom evaluation (simple split since SetFit trainer handles it internal if we pass eval_dataset, 
        # but let's do a manual split for simplicity/control over the reported metric in metadata)
        from sklearn.model_selection import train_test_split
        import time
        from datetime import datetime
        import json

        start_time = time.time()

        # Simple 80/20 split
        # If we have very few examples, this might be fragile, but it's an improvement.
        if len(examples) >= 5:
            train_ex, test_ex = train_test_split(examples, test_size=0.2, random_state=42)
        else:
            train_ex = examples
            test_ex = examples # fallback to testing on train (not ideal but better than crash)

        train_data = {"text": [e.text for e in train_ex], "label": [e.label for e in train_ex]}
        test_data = {"text": [e.text for e in test_ex], "label": [e.label for e in test_ex]}
        
        train_dataset = Dataset.from_dict(train_data)
        test_dataset = Dataset.from_dict(test_data)

        # Load base model
        model = SetFitModel.from_pretrained(base_model_name)

        # Create training arguments
        args = TrainingArguments(
            batch_size=batch_size,
            num_epochs=1,
            num_iterations=num_iterations,
            body_learning_rate=learning_rate,
            sampling_strategy="oversampling" 
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            metric="accuracy"
        )

        # Train
        logger.info("Training...")
        trainer.train()
        
        # Evaluate
        metrics = trainer.evaluate()
        accuracy = metrics.get("accuracy", 0.0)
        logger.info(f"Training complete. Accuracy: {accuracy}")

        duration = time.time() - start_time
        
        # Save model
        save_path = self._get_model_path(model_name)
        logger.info(f"Saving model to {save_path}")
        model.save_pretrained(save_path)
        
        # Save Metadata
        metadata = {
            "model_name": model_name,
            "base_model": base_model_name,
            "created_at": datetime.utcnow().isoformat(),
            "accuracy": accuracy,
            "training_duration_seconds": duration,
            "num_examples": len(examples)
        }
        with open(os.path.join(save_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # Cache loaded model
        self.loaded_models[model_name] = model


    def predict(self, model_name: str, texts: List[str]):
        model = self.load_model(model_name)
        preds = model.predict(texts)
        # SetFit predict returns a numpy array or list of labels directly? 
        # It usually returns just the labels if fit on string labels? 
        # But let's check. Actually SetFit predict returns the labels. 
        # But we might want scores. 
        # model.predict_proba(texts) gives probabilities.
        
        return preds.tolist() # Convert numpy array to list

    def get_embeddings(self, model_name: str, texts: List[str], dimensions: int = None):
        model = self.load_model(model_name)
        # SetFitModel wraps a SentenceTransformer body
        embeddings = model.model_body.encode(texts)
        
        if dimensions:
             # Matryoshka slicing or truncation
             embeddings = embeddings[:, :dimensions]
             
        return embeddings.tolist()
