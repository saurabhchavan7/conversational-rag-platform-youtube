# test_api.py
from youtube_transcript_api import YouTubeTranscriptApi

video_id = "O5xeyoRL95U"
api = YouTubeTranscriptApi()

print("Testing with instance...")
try:
    transcript = api.fetch(video_id, languages=["en"])
    print(f"SUCCESS! Got {len(transcript)} segments")
    print(f"First segment: {transcript[0]}")
except Exception as e:
    print(f"FAILED: {e}")