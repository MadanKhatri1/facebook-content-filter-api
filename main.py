from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from groq import Groq
import uvicorn

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_text(request: Request):
    data = await request.json()
    post_text = data.get("post", "")
    comment_text = data.get("text", "")

    print(f"Analyzing comment: {comment_text} on post: {post_text}")

    
    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a strict classifier. Consider words like 'Rip sale' or 'sale' as negative and toxic.Reply with ONLY one word: toxic or non-toxic."
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
        sentiment = "toxic"
        hide = True
    elif response_text == "non-toxic":
        sentiment = "non-toxic"
        hide = False
    else:
        # fallback if model outputs something unexpected
        sentiment = response_text
        hide = False


    return {"sentiment": sentiment, "hide": hide}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
