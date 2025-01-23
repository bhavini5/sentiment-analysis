from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, Pipeline
from mangum import Mangum  # For AWS Lambda integration
import logging

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize sentiment analysis pipeline
try:
    logger.info("Loading sentiment analysis model...")
    sentiment_pipeline: Pipeline = pipeline("sentiment-analysis")
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error("Error loading sentiment analysis model: %s", str(e))
    raise RuntimeError("Failed to load sentiment analysis model.") from e

# Define input data model
class SentimentRequest(BaseModel):
    text: str

# Root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to the Sentiment Analysis API"}

# Sentiment analysis endpoint
@app.post("/analyze-sentiment")
def analyze_sentiment(request: SentimentRequest):
    try:
        logger.info("Received request: %s", request.text)
        result = sentiment_pipeline(request.text)
        logger.info("Analysis result: %s", result)
        return {
            "text": request.text,
            "sentiment": result[0]["label"],
            "score": result[0]["score"],
        }
    except Exception as e:
        logger.error("Error processing sentiment analysis: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Add Mangum handler for AWS Lambda
handler = Mangum(app)