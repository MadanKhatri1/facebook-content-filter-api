from fastapi import FastAPI, Request, HTTPException, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()

# Security setup
API_KEY = os.getenv("API_KEY")  # We'll set this in Render
if not API_KEY:
    raise ValueError("API_KEY environment variable not set")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()

# CORS still needed so browser extension can call it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to chrome-extension://*
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Check if the provided API key matches our secret"""
    if api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API key. Please subscribe to use this service."
        )
    return api_key

@app.post("/analyze")
async def analyze_text(request: Request, api_key: str = Security(verify_api_key)):
    """
    Analyze comment toxicity.
    Requires X-API-Key header.
    """
    data = await request.json()
    post_text = data.get("post", "")
    comment_text = data.get("text", "")

    print(f"Analyzing comment: {comment_text[:50]}... on post: {post_text[:50]}...")

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict classifier. Consider words like 'Rip sale' or 'sale' as negative and toxic. Reply with ONLY one word: toxic or non-toxic."
                },
                {
                    "role": "user",
                    "content": f"Post: {post_text}\nComment: {comment_text}"
                }
            ],
            temperature=0,
            max_tokens=3
        )

        response_text = completion.choices[0].message.content.lower().strip()
        print(f"Model response: {response_text}")

        if response_text == "toxic":
            return {"sentiment": "toxic", "hide": True}
        elif response_text == "non-toxic":
            return {"sentiment": "non-toxic", "hide": False}
        else:
            return {"sentiment": response_text, "hide": False}
            
    except Exception as e:
        print(f"Error calling Groq: {e}")
        raise HTTPException(status_code=500, detail="Analysis service temporarily unavailable")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "alive", "service": "Facebook Content Filter API"}

# Remove the if __name__ == "__main__" block - Render handles this