from fastapi import FastAPI, File, UploadFile, HTTPException
from model import VectorDatabase
import os

# Parse environment variables for embeddings configuration and MLflow experiment
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
EMBEDDINGS_ENDPOINT = os.getenv("EMBEDDINGS_ENDPOINT", "http://localhost:8000")
MODEL_NAME = os.getenv("EMBEDDINGS_MODEL_NAME", "nvidia/nv-embedqa-e5-v5")

if not EXPERIMENT_NAME or not NVIDIA_API_KEY or not EMBEDDINGS_ENDPOINT or not MODEL_NAME:
    raise RuntimeError("Missing required environment variables: MLFLOW_EXPERIMENT_NAME, NVIDIA_API_KEY, EMBEDDINGS_ENDPOINT, EMBEDDINGS_MODEL_NAME")

# Initialize the vector database
vector_db = VectorDatabase(EXPERIMENT_NAME, NVIDIA_API_KEY, EMBEDDINGS_ENDPOINT, MODEL_NAME)

app = FastAPI()

@app.get("/retrieve")
async def retrieve(query: str, top_k: int = 5):
    """Retrieve top_k results for a given query."""
    try:
        results = vector_db.retrieve(query, top_k)
        return {"query": query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add")
async def add(file: UploadFile = File(...)):
    """Add a new document (e.g., a PDF) to the vector database."""
    try:
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        vector_db.add_document(file_path)
        return {"message": f"File {file.filename} added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
