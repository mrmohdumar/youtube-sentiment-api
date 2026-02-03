from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from typing import List
import os

# Initialize app
app = FastAPI(
    title="YouTube Sentiment Analysis API",
    description="DistilBERT fine-tuned on YouTube comments",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
tokenizer = None
id2label = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

# Load model on startup
@app.on_event("startup")
async def load_model():
    global model, tokenizer
    print("Loading model...")
    
    MODEL_PATH = "./youtube_sentiment_model"
    
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model directory not found at {MODEL_PATH}")
        return
    
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()
        print("✅ Model loaded successfully!")
        print(f"   Accuracy: 70.71%")
        print(f"   Baseline: 51.80% (VADER)")
    except Exception as e:
        print(f"ERROR loading model: {e}")

# Request models
class CommentRequest(BaseModel):
    text: str

class CommentBatchRequest(BaseModel):
    comments: List[str]

# Root endpoint
@app.get("/")
def root():
    return {
        "message": "YouTube Sentiment Analysis API",
        "status": "running" if model is not None else "model_not_loaded",
        "model": "DistilBERT-base-uncased (fine-tuned)",
        "accuracy": "70.71%",
        "baseline": "51.80% (VADER)",
        "improvement": "+18.91%",
        "endpoints": {
            "/analyze": "Analyze single comment",
            "/analyze_batch": "Analyze multiple comments",
            "/health": "Health check"
        }
    }

# Health check
@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None
    }

# Analyze single comment
@app.post("/analyze")
def analyze(request: CommentRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        inputs = tokenizer(
            request.text,
            return_tensors='pt',
            truncation=True,
            max_length=128,
            padding='max_length'
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            pred = probs.argmax().item()
        
        return {
            "text": request.text,
            "sentiment": id2label[pred],
            "confidence": float(probs[pred]),
            "probabilities": {
                "negative": float(probs[0]),
                "neutral": float(probs[1]),
                "positive": float(probs[2])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Analyze batch
@app.post("/analyze_batch")
def analyze_batch(request: CommentBatchRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        if not request.comments:
            raise HTTPException(status_code=400, detail="No comments provided")
        
        if len(request.comments) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 comments per request")
        
        # Filter empty comments
        valid_comments = [c.strip() for c in request.comments if c.strip()]
        
        if not valid_comments:
            raise HTTPException(status_code=400, detail="No valid comments after filtering")
        
        # Tokenize
        inputs = tokenizer(
            valid_comments,
            return_tensors='pt',
            truncation=True,
            max_length=128,
            padding=True
        )
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = probs.argmax(dim=1)
        
        # Format results
        results = []
        for i, comment in enumerate(valid_comments):
            results.append({
                "text": comment,
                "sentiment": id2label[preds[i].item()],
                "confidence": float(probs[i][preds[i]]),
                "probabilities": {
                    "negative": float(probs[i][0]),
                    "neutral": float(probs[i][1]),
                    "positive": float(probs[i][2])
                }
            })
        
        # Calculate summary
        counts = {
            "positive": sum(1 for r in results if r["sentiment"] == "Positive"),
            "negative": sum(1 for r in results if r["sentiment"] == "Negative"),
            "neutral": sum(1 for r in results if r["sentiment"] == "Neutral")
        }
        
        total = len(results)
        
        return {
            "results": results,
            "summary": {
                "total": total,
                "distribution": {
                    "positive": round(counts["positive"] / total * 100, 1),
                    "negative": round(counts["negative"] / total * 100, 1),
                    "neutral": round(counts["neutral"] / total * 100, 1)
                },
                "counts": counts
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))