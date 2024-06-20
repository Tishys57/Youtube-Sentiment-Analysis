import csv
import os
import joblib
import googleapiclient.discovery
from collections import Counter
import html

# Replace 'YOUR_API_KEY_HERE' with your actual YouTube Data API key
API_KEY = 'AIzaSyD_hodyDC5UDFoJrlcFQs0EeLq2E93VR-M'

def get_comments(video_id):
    # Disable OAuthlib's HTTPS verification when running locally.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=API_KEY)

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100
    )

    comments = []

    while request:
        response = request.execute()
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comment = html.unescape(comment)  # Decode HTML entities
            comments.append([comment])
        
        if 'nextPageToken' in response:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=response['nextPageToken']
            )
        else:
            request = None

    return comments

def write_comments_to_csv(comments, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Comment', 'Sentiment'])
        writer.writerows(comments)

def predict_sentiment(comments, model):
    # Assuming the model expects a list of comment texts for prediction
    texts = [comment[0] for comment in comments]
    sentiments = model.predict(texts)
    comments_with_sentiment = [(comments[i][0], sentiment) for i, sentiment in enumerate(sentiments)]
    # Return unsorted comments with sentiment for immediate display
    return comments_with_sentiment

def calculate_sentiment_percentage(comments_with_sentiment):
    sentiments = [comment[1] for comment in comments_with_sentiment]
    sentiment_counts = Counter(sentiments)

    total = sum(sentiment_counts.values())
    positive = sentiment_counts.get('positive', 0)
    neutral = sentiment_counts.get('neutral', 0)
    negative = sentiment_counts.get('negative', 0)

    if total == 0:
        return "No comments to analyze."

    positive_percentage = (positive / total) * 100
    neutral_percentage = (neutral / total) * 100
    negative_percentage = (negative / total) * 100

    return {
        "positive_percentage": positive_percentage,
        "neutral_percentage": neutral_percentage,
        "negative_percentage": negative_percentage
    }

if __name__ == "__main__":
    video_url = input("Enter the YouTube video URL: ")

    # Extract the video ID from the URL
    if 'v=' in video_url:
        video_id = video_url.split('v=')[1]
        ampersand_position = video_id.find('&')
        if ampersand_position != -1:
            video_id = video_id[:ampersand_position]
    else:
        print("Invalid YouTube URL")
        exit()

    # Load the comments from the YouTube video
    comments = get_comments(video_id)

    # Load the logistic regression sentiment analysis model
    model = joblib.load('logistic_regression_model.joblib')

    # Predict the sentiment of the comments and return unsorted
    comments_with_sentiment = predict_sentiment(comments, model)

    # Write the comments with sentiment to a CSV file
    write_comments_to_csv(comments_with_sentiment, "comments_with_sentiment.csv")
    print("Comments with sentiment have been written to comments_with_sentiment.csv")

    # Calculate and print the sentiment percentage
    sentiment_percentage = calculate_sentiment_percentage(comments_with_sentiment)
    print(f"Sentiment Percentages - Positive: {sentiment_percentage['positive_percentage']:.2f}%, Neutral: {sentiment_percentage['neutral_percentage']:.2f}%, Negative: {sentiment_percentage['negative_percentage']:.2f}%")
