from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from fastapi.middleware.cors import CORSMiddleware
import os
import gdown

# Directory to store model files
model_dir = "model"

# Public Google Drive links
files_to_download = {
    "model.safetensors": "https://drive.google.com/uc?id=1QV1Atx7b5gJGPjCyUDJLN6AxAevyNQ4C",
    "config.json": "https://drive.google.com/uc?id=1sAHtDD-sUpT2DCkfHrl2-heLFmxRhrR6",
    "tokenizer_config.json": "https://drive.google.com/uc?id=15NpYYfPw5MkoyAACRiM2Gyz4wocZJF7j",
    "vocab.txt": "https://drive.google.com/uc?id=1JhKnVV9AomMMyKtdFgsKeC6df2grdF_K",
    "special_tokens_map.json": "https://drive.google.com/uc?id=1SycDz4PuiPZI8E3lv-QqYw-dGC1Tcptl",
}

# Function to download a file from Google Drive
def download_file_from_gdrive(url, destination):
    if not os.path.exists(destination):
        print(f"Downloading {destination} ...")
        gdown.download(url, destination, quiet=False)

# Check if model is already downloaded
def model_already_downloaded():
    return all(os.path.exists(os.path.join(model_dir, file)) for file in files_to_download)

# Download model files
def download_model():
    os.makedirs(model_dir, exist_ok=True)
    for filename, url in files_to_download.items():
        filepath = os.path.join(model_dir, filename)
        download_file_from_gdrive(url, filepath)

# Only download if not already downloaded
if not model_already_downloaded():
    download_model()

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Human-readable label mapping
label_map = {
    "LABEL_0": "Non-Toxic",
    "LABEL_1": "Negative",
    "LABEL_2": "Toxic"
}

# FastAPI app initialization
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schemas
class InputText(BaseModel):
    inputs: str

class InputBatch(BaseModel):
    inputs: list[str]

# Root route (health check)
@app.get("/")
async def root():
    return {"message": "Sentiment analyzer is running!"}

# Single input endpoint
@app.post("/analyze")
async def analyze_sentiment(data: InputText):
    try:
        result = classifier(data.inputs)[0]
        readable_label = label_map.get(result["label"], result["label"])
        return {"label": readable_label, "score": float(result["score"])}
    except Exception as e:
        return {"error": str(e)}

# Batch input endpoint
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

# Render-compatible entry point
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Render dynamically assigns port
    uvicorn.run("main:app", host="0.0.0.0", port=port)
