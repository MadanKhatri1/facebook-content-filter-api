from fastapi import FastAPI, Request, HTTPException, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not set")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

async def verify_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid key")
    return api_key

@app.post("/analyze")
async def analyze(request: Request, api_key: str = Security(verify_key)):
    data = await request.json()
    text = data.get("text", "")
    
    if not text or len(text) < 2:
        return {"sentiment": "neutral", "hide": False}
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system", 
                    "content": "Detect toxic/hateful/spam comments in Nepali (नेपाली), English, or mixed. Reply ONLY with: toxic OR safe. Be strict with insults like 'mula', 'sale', 'randi'."
                },
                {"role": "user", "content": f"Comment: {text}"}
            ],
            temperature=0,
            max_tokens=5
        )
        
        result = completion.choices[0].message.content.lower().strip()
        is_toxic = "toxic" in result
        
        return {
            "sentiment": "toxic" if is_toxic else "non-toxic",
            "hide": is_toxic,
            "confidence": 0.95
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return {"sentiment": "non-toxic", "hide": False}

@app.get("/")
async def health():
    return {"status": "alive", "service": "Nepali-Content-Filter", "model": "Groq-LLaMA"}
