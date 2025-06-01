from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from fastapi.middleware.cors import CORSMiddleware
import os
import gdown

# Directory to store model files
model_dir = "model"

# URLs to download model files from Google Drive (public sharing links)
files_to_download = {
    "model.safetensors": "https://drive.google.com/uc?id=1QV1Atx7b5gJGPjCyUDJLN6AxAevyNQ4C",
    "config.json": "https://drive.google.com/uc?id=1sAHtDD-sUpT2DCkfHrl2-heLFmxRhrR6",
    "tokenizer_config.json": "https://drive.google.com/uc?id=15NpYYfPw5MkoyAACRiM2Gyz4wocZJF7j",
    "vocab.txt": "https://drive.google.com/uc?id=1JhKnVV9AomMMyKtdFgsKeC6df2grdF_K",
    "special_tokens_map.json": "https://drive.google.com/uc?id=1SycDz4PuiPZI8E3lv-QqYw-dGC1Tcptl",
}

def download_file_from_gdrive(url, destination):
    if not os.path.exists(destination):
        print(f"Downloading {destination} ...")
        gdown.download(url, destination, quiet=False)

def model_already_downloaded():
    return all(os.path.exists(os.path.join(model_dir, file)) for file in files_to_download)

def download_model():
    os.makedirs(model_dir, exist_ok=True)
    for filename, url in files_to_download.items():
        filepath = os.path.join(model_dir, filename)
        download_file_from_gdrive(url, filepath)

# Download model files if not already present
if not model_already_downloaded():
    download_model()

# Load tokenizer and model from local directory
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Label mapping
label_map = {
    "LABEL_0": "Non-Toxic",
    "LABEL_1": "Negative",
    "LABEL_2": "Toxic"
}

# FastAPI app
app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body schemas
class InputText(BaseModel):
    inputs: str

class InputBatch(BaseModel):
    inputs: list[str]

# Root route (health check)
@app.get("/")
async def root():
    return {"message": "Sentiment analyzer is running!"}

# Single input analysis
@app.post("/analyze")
async def analyze_sentiment(data: InputText):
    try:
        result = classifier(data.inputs)[0]
        readable_label = label_map.get(result["label"], result["label"])
        return {"label": readable_label, "score": float(result["score"])}
    except Exception as e:
        return {"error": str(e)}

# Batch input analysis
@app.post("/analyze_batch")
async def analyze_batch(data: InputBatch):
    try:
        results = classifier(data.inputs)
        return [
            {
                "label": label_map.get(r["label"], r["label"]),
                "score": float(r["score"])
            } for r in results
        ]
    except Exception as e:
        return {"error": str(e)}
