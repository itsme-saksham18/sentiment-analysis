from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from fastapi.middleware.cors import CORSMiddleware

# Load model and tokenizer from the 'model' folder
model_dir = "model"
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
