# Use PyTorch image with CUDA support (devel needed for flash-attn compilation)
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

# prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive


WORKDIR /app

COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Create directory for models
RUN mkdir -p /app/models

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
