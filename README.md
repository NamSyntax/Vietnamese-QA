# Vietnamese-QA

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/transformers-HuggingFace-orange.svg)](https://huggingface.co/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688.svg)](https://fastapi.tiangolo.com/)

**Vietnamese-QA** is an end-to-end Machine Learning pipeline for Vietnamese Question Answering. The project leverages **XLM-RoBERTa Large**, meticulously fine-tuned on the [UIT-ViQuAD 2.0](https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0) dataset to deliver precise, context-aware answers.

> **Pre-trained Model Available:** You can immediately experience or integrate our fine-tuned weights hosted directly on Hugging Face: [NamSyntax/xlmr-large-viquad](https://huggingface.co/NamSyntax/xlmr-large-viquad).

---

## Key Features
- **Clean Architecture:** Modular design (`src/`, `configs/`) focused on scalability.
- **Config-Driven:** Easily tune hyperparameters and execution behaviors via `config.yaml`.
- **Gradio Web UI:** Built-in interactive frontend for quick demonstrations and qualitative testing.
- **FastAPI Integration:** Lightweight, blazing-fast production-ready API for deployment.
- **Containerization:** Out-of-the-box Docker support for seamless scaling.

---

## Installation

I recommend using a modern package manager like `uv` or standard `pip`. Ensure you have Python 3.10+ installed.

```bash
# Clone the repository
git clone https://github.com/NamSyntax/Vietnamese-QA.git
cd Vietnamese-QA

# Setup a virtual environment (using uv)
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
uv pip install pyyaml fastapi uvicorn pydantic
```

---

## Usage Guide

The program is fully controlled via the central `main.py` entrypoint using the `--mode` flag. Settings and hyperparameters are managed via `configs/config.yaml`.

### 1. Model Training
To start fine-tuning the model on the ViQuAD dataset:
```bash
python main.py --mode train
```

### 2️. Model Evaluation
To evaluate the model's performance (Metric: SQuAD V2 / Exact Match & F1) on the validation set:
```bash
python main.py --mode eval
```

### 3️. Interactive UI (Gradio)
Launch a web-based testing interface to ask your own questions:
```bash
python main.py --mode gradio
```

### 4️. REST API Server (FastAPI)
Deploy the model as a production-grade backend service:
```bash
python main.py --mode api
```

#### API Endpoint Example
Send a POST request to the `/predict` endpoint:
```bash
curl -X POST "http://0.0.0.0:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"context": "Hà Nội là thủ đô của nước Cộng hòa Xã hội chủ nghĩa Việt Nam...", "question": "Thủ đô của Việt Nam là gì?"}'
```

---

## 🐳 Docker Deployment

Easily containerize the application for any environment (requires Docker installed):

```bash
# Step 1: Build the Docker Image
docker build -t vietnamese-qa-api .

# Step 2: Run the container (Mapping port 8000)
docker run -p 8000:8000 vietnamese-qa-api
```
*Access the Swagger API Documentation at: `http://localhost:8000/docs`*

---

## Project Structure

```text
Vietnamese-QA/
│
├── configs/
│   └── config.yaml           # Hyperparameters & App configuration
├── src/
│   ├── api.py                # FastAPI endpoints and model serving
│   ├── data_loader.py        # HuggingFace Datasets loading logic
│   ├── evaluate.py           # Validation and metric evaluation
│   ├── inference.py          # Gradio Interactive UI
│   ├── preprocessing.py      # Text tokenization & chunking logic
│   ├── train.py              # HF Trainer pipeline logic
│   └── utils.py              # Post-processing and SQuAD metrics
│
├── main.py                   # Central CLI entry point
├── Dockerfile                # Recipe for API containerization
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

