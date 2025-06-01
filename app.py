from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from fastapi.middleware.cors import CORSMiddleware
import os
import requests

# Directory to store model files
model_dir = "model"

# URLs to download model files from Google Drive (replace with your actual file IDs)
files_to_download = {
    "model.safetensors": "https://drive.google.com/uc?export=download&id=1QV1Atx7b5gJGPjCyUDJLN6AxAevyNQ4C",
    "config.json": "https://drive.google.com/uc?export=download&id=1sAHtDD-sUpT2DCkfHrl2-heLFmxRhrR6",
    "tokenizer_config.json": "https://drive.google.com/uc?export=download&id=15NpYYfPw5MkoyAACRiM2Gyz4wocZJF7j",
    "vocab.txt": "https://drive.google.com/uc?export=download&id=1JhKnVV9AomMMyKtdFgsKeC6df2grdF_K",
    "special_tokens_map.json": "https://drive.google.com/uc?export=download&id=1SycDz4PuiPZI8E3lv-QqYw-dGC1Tcptl",
    # Add other files your model needs here
}

def download_file_from_gdrive(url, destination):
    if not os.path.exists(destination):
        print(f"Downloading {destination} ...")
        response = requests.get(url)
        response.raise_for_status()
        with open(destination, "wb") as f:
            f.write(response.content)

def download_model():
    os.makedirs(model_dir, exist_ok=True)
    for filename, url in files_to_download.items():
        filepath = os.path.join(model_dir, filename)
        download_file_from_gdrive(url, filepath)

# Download model files before loading
download_model()

# Load tokenizer and model from local directory
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body
class InputText(BaseModel):
    inputs: str

@app.post("/analyze")
async def analyze_sentiment(data: InputText):
    result = classifier(data.inputs)
    return {"label": result[0]["label"], "score": float(result[0]["score"])}
