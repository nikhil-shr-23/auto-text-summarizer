from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import pipeline
import torch
import os


# Initialize FastAPI app
app = FastAPI()

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


MODEL_NAME = "philschmid/distilbart-cnn-12-6-samsum"

summarizer = None

def get_summarizer():
    global summarizer
    if summarizer is None:
        try:
            summarizer = pipeline(
                "summarization",
                model=MODEL_NAME,
                device=-1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else None
            )
        except Exception as e:
            print(f"Model loading failed: {str(e)}")
            return None
    return summarizer

class SummaryRequest(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/summarize")
async def summarize(request: SummaryRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Please enter some text.")
        
    if len(request.text) > 10000:
        raise HTTPException(status_code=400, detail="Text is too long. Please limit to 10,000 characters.")
    
    summarizer = get_summarizer()    
    if summarizer is None:
        raise HTTPException(status_code=500, detail="Model not properly loaded.")
        
    try:
        summary = summarizer(
            request.text.strip(), 
            max_length=130, 
            min_length=30, 
            do_sample=False
        )
        
        return {
            "summary_text": summary[0]['summary_text'],
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}") 
    
