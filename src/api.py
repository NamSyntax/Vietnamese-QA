from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import yaml
import uvicorn
import os

app = FastAPI(
    title="Vietnamese QA API",
    description="API cho mô hình nhận diện câu trả lời tiếng Việt (XLM-RoBERTa)",
    version="1.0.0"
)

# Global variable to hold the pipeline instance
qa_pipe = None

class QARequest(BaseModel):
    """Input schema for QA request."""
    context: str
    question: str

class QAResponse(BaseModel):
    """Output schema for system response."""
    answer: str
    score: float

@app.on_event("startup")
def load_model():
    """Loads configuration and model from the specified path upon server startup."""
    global qa_pipe
    config_path = os.getenv("CONFIG_PATH", "configs/config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ckpt_path = config["model"]["ckpt_path"]
    print(f"Loading model from {ckpt_path}...")
    
    try:
        # Load components and initialize the QA pipeline
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=True)
        model = AutoModelForQuestionAnswering.from_pretrained(ckpt_path)
        qa_pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.post("/predict", response_model=QAResponse)
def predict(request: QARequest):
    """Receives context and question, returns the model's answer and confidence score."""
    if qa_pipe is None:
        raise HTTPException(status_code=503, detail="Model is not ready or failed to load.")
    
    try:
        # Execute prediction pipeline
        out = qa_pipe({"context": request.context, "question": request.question})
        return QAResponse(answer=out["answer"], score=out["score"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_api(config: dict):
    """Starts the FastAPI application based on host and port specifications."""
    host = config.get("inference", {}).get("api_host", "0.0.0.0")
    port = config.get("inference", {}).get("api_port", 8000)
    uvicorn.run("src.api:app", host=host, port=port, reload=False)
