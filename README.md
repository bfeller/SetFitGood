# SetFit Docker API

A self-hosted, Dockerized API for training and serving [SetFit](https://github.com/huggingface/setfit) models for efficient few-shot text classification.

## Features

- **REST API**: Built with FastAPI.
- **Train**: Fine-tune models on your own data via a simple JSON endpoint.
- **Predict**: specific intent classification with high accuracy from few examples.
- **Persistence**: Trained models are saved to disk and persisted across restarts.

## Local Development

### Prerequisites

- Docker
- Docker Compose

### Hugging Face Authentication (Optional)

If you need to access private repositories, gated models (like Llama 3 or GTE-ModernBERT), or avoid rate limits, you must provide your Hugging Face User Access Token.

1.  Get your token from [Hugging Face Settings](https://huggingface.co/settings/tokens).
2.  Export it as an environment variable in your terminal:
    ```bash
    export HF_TOKEN=hf_your_token_here
    ```
    *Note: The `docker-compose.yml` is configured to pass this variable through to the container.*

### Running the API

1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd <repo_name>
   ```

2. Start the service:
   ```bash
   docker compose up --build -d
   ```

3. The API will be available at `http://localhost:8000`. You can view the interactive documentation at `http://localhost:8000/docs`.

## API Reference

### Authentication
If the `API_KEY` environment variable is set, all requests (except `/health`) must include the header:
`X-API-Key: <your_key>`

**Helper Command**:
To generate a secure random API key, you can run:
```bash
openssl rand -hex 32
```

### 1. Health Check
**GET** `/health`
- **Response**: `{"status": "ok"}`

### 2. Train Model
**POST** `/v1/train`
Starts an asynchronous training job.

**Request Body**:
```json
{
  "model_name": "string (unique identifier)",
  "base_model": "string (default: Alibaba-NLP/gte-modernbert-base)",
  "examples": [
    {
      "text": "string",
      "label": "string"
    }
  ],
  "num_iterations": "int (default: 20)",
  "learning_rate": "float (default: 2e-5)",
  "batch_size": "int (default: 16)"
}
```

**Response**:
```json
{
  "job_id": "uuid",
  "status": "pending",
  "model_name": "string",
  "created_at": "timestamp"
}
```

### 3. Get Job Status
**GET** `/v1/jobs/{job_id}`
Check the status of a training job.

**Response**:
```json
{
  "job_id": "uuid",
  "status": "pending | running | completed | failed",
  "model_name": "string",
  "created_at": "timestamp",
  "error": "string | null"
}
```

### 4. Get Model Metadata
**GET** `/v1/models/{model_name}`
Retrieve training metrics and metadata for a completed model.

**Response**:
```json
{
  "model_name": "string",
  "base_model": "string",
  "created_at": "iso_timestamp",
  "accuracy": "float",
  "training_duration_seconds": "float",
  "num_examples": "int"
}
```

### 5. Predict Intent
**POST** `/v1/predict`
Get classification predictions for a list of texts.

**Request Body**:
```json
{
  "model_name": "string",
  "texts": ["string"]
}
```

**Response**:
```json
{
  "predictions": [
    {
      "text": "string",
      "label": "string",
      "score": "float | null"
    }
  ]
}
```

### 6. Generate Embeddings
**POST** `/v1/embeddings`
Get vector embeddings for a list of texts using the underlying sentence transformer.

**Request Body**:
```json
{
  "model_name": "string",
  "texts": ["string"]
}
```

**Response**:
```json
{
  "embeddings": [
    [0.123, -0.456, ...] // Vector of length 768 (for default model)
  ]
}
```


## Model Selection & Compatibility

The efficacy of SetFit depends heavily on the underlying **base model** used for generating embeddings.

### Changing the Base Model
You can use any compatible model by specifying the `base_model` parameter in your `/v1/train` request.

### Supported Models
The API supports any model compatible with the [Sentence Transformers](https://sbert.net/) library. This covers thousands of models on the Hugging Face Hub.

**Recommended Models:**
- **`Alibaba-NLP/gte-modernbert-base`** (Default): Excellent all-rounder, supports Flash Attention 2, 8k context window.
- **`BAAI/bge-base-en-v1.5`**: Strong performance on retrieval tasks.
- **`sentence-transformers/paraphrase-mpnet-base-v2`**: reliable, classic choice.
- **`intfloat/e5-large-v2`**: High quality embeddings.

**Note on Embeddings**:
The vector length returned by `/v1/embeddings` changes based on the selected model:
- `*-base` models usually have **768** dimensions.
- `*-large` models usually have **1024** dimensions.
- `*-MiniLM-*` models usually have **384** dimensions.

## Deployment

### Coolify (Self-Hosted)
[Coolify](https://coolify.io/) is a recommended platform for self-hosting.

1. **Create Resource**: Add a "Git Repository" project pointing to this repo.
2. **Build Pack**: Select `Docker Compose`.
3. **Environment**:
   - `MODEL_PATH`: `/app/models` (Recommended)
   - `API_KEY`: Set this to secure your API.
4. **Volumes**:
   - Map a persistent volume to `/app/models` to save your trained models.
5. **GPU Support**: 
   - Ensure your Coolify server has NVIDIA drivers.
   - The `docker-compose.yml` is configured to request 1 GPU.

### Standard Docker
```bash
docker compose up --build -d
```

## License

MIT License. See [LICENSE](LICENSE) for details.
