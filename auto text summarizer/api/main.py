from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import pipeline
import os
from mangum import Mangum  # Required for Vercel

app = FastAPI()

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


MODEL_NAME = "sshleifer/distilbart-cnn-6-6"


@app.on_event("startup")
def load_model():
    global summarizer
    try:
        summarizer = pipeline(
            "summarization",
            model=MODEL_NAME,
            device=-1,
            torch_dtype="auto"
        )
    except Exception as e:
        print(f"Model load failed: {str(e)}")
        summarizer = None

class SummaryRequest(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/summarize")
async def summarize(request: SummaryRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")
    
    if not summarizer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        summary = summarizer(
            request.text.strip(),
            max_length=130,
            min_length=30,
            do_sample=False
        )
        return {"summary_text": summary[0]['summary_text']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


handler = Mangum(app)
