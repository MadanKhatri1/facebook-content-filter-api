from fastapi import FastAPI, Request, HTTPException, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

load_dotenv()

# Security setup
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable not set")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

print("🤖 Loading multilingual model... This takes 30 seconds...")
try:
    
    MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment-latest"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()  # Disable training mode (saves memory)
    
    # Map sentiment to toxicity (negative sentiment = likely toxic)
    id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    print("✅ Model loaded! Ready for Nepali + English text.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    raise

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_allow_headers=["*"],
)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

@app.post("/analyze")
async def analyze_text(request: Request, api_key: str = Security(verify_api_key)):
    data = await request.json()
    post_text = data.get("post", "")
    comment_text = data.get("text", "")
    
    if not comment_text or len(comment_text) < 2:
        return {"sentiment": "neutral", "hide": False}
    
    print(f"Analyzing: {comment_text[:50]}...")
    
    try:
        inputs = tokenizer(
            comment_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        )
        
        # ✅ Inference without gradients (faster, less memory)
        with torch.no_grad():
            outputs = model(**inputs)
            scores = outputs.logits[0].numpy()
            predicted_class = np.argmax(scores)
            label = id2label[predicted_class]
            confidence = float(np.exp(scores[predicted_class]) / np.sum(np.exp(scores)))
        
        print(f"Result: {label} ({confidence:.2f})")
        
        is_toxic = (label == 'negative' and confidence > 0.6)
        
        return {
            "sentiment": "toxic" if is_toxic else "non-toxic",
            "hide": is_toxic,
            "confidence": round(confidence, 2),
            "label": label
        }
            
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed")

@app.get("/")
async def root():
    return {
        "status": "alive", 
        "model": "multilingual-sentiment",
        "languages": "Nepali, English, Hindi, 100+"
    }