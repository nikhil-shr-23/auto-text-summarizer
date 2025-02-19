from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import pipeline
import torch
import os

# Initialize FastAPI app
app = FastAPI()

# Get absolute path to templates directory
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Initialize the summarization pipeline with lazy loading
summarizer = None

def get_summarizer():
    global summarizer
    if summarizer is None:
        try:
            summarizer = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",
                device=-1  # Force CPU usage on Vercel
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return summarizer

class SummaryRequest(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        print(f"Error rendering template: {str(e)}")
        return HTMLResponse(content=f"Error: {str(e)}", status_code=500)

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