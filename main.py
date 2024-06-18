from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse
import uvicorn
import joblib
import os
from pathlib import Path
from get_comments import get_comments, predict_sentiment, write_comments_to_csv, calculate_sentiment_percentage

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the index.html file
@app.get("/", response_class=HTMLResponse)
async def read_index():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(content=html_path.read_text(), status_code=200)

# Endpoint to analyze YouTube comments
@app.post("/analyze")
async def analyze_comments(request: Request):
    data = await request.json()
    video_url = data.get("video_url")
    if not video_url:
        return JSONResponse(content={"error": "No URL provided"}, status_code=400)
    
    # Extract the video ID from the URL
    if 'v=' in video_url:
        video_id = video_url.split('v=')[1]
        ampersand_position = video_id.find('&')
        if ampersand_position != -1:
            video_id = video_id[:ampersand_position]
    else:
        return JSONResponse(content={"error": "Invalid YouTube URL"}, status_code=400)
    
    # Load the comments from the YouTube video
    comments = get_comments(video_id)
    
    # Load the logistic regression sentiment analysis model
    model = joblib.load('logistic_regression_model.joblib')
    
    # Predict the sentiment of the comments
    comments_with_sentiment = predict_sentiment(comments, model)
    
    # Write the comments with sentiment to a CSV file
    write_comments_to_csv(comments_with_sentiment, "comments_with_sentiment.csv")
    
    # Calculate the sentiment percentage
    sentiment_percentage = calculate_sentiment_percentage(comments_with_sentiment)
    
    response = {
        "sentiment_percentage": sentiment_percentage,
        "comments": comments_with_sentiment
    }
    
    return JSONResponse(content=response)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
