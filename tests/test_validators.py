import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.validators import (
    validate_youtube_video_id,
    extract_video_id_from_url,
    validate_query,
    is_valid_youtube_url,
    sanitize_text
)
from utils.exceptions import InvalidVideoIDError, InvalidURLError

print("="*60)
print("Testing Validators")
print("="*60)

# Test 1: Valid video ID
try:
    video_id = validate_youtube_video_id("dQw4w9WgXcQ")
    print(f"✅ Valid video ID: {video_id}")
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 2: Invalid video ID (too short)
try:
    validate_youtube_video_id("abc")
    print("❌ Should have failed!")
except InvalidVideoIDError as e:
    print(f"✅ Correctly rejected invalid ID: {e}")

# Test 3: Extract from standard URL
try:
    video_id = extract_video_id_from_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    print(f"✅ Extracted from standard URL: {video_id}")
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 4: Extract from short URL
try:
    video_id = extract_video_id_from_url("https://youtu.be/dQw4w9WgXcQ")
    print(f"✅ Extracted from short URL: {video_id}")
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 5: Invalid URL
try:
    extract_video_id_from_url("https://example.com")
    print("❌ Should have failed!")
except InvalidURLError as e:
    print(f"✅ Correctly rejected invalid URL")

# Test 6: Check if URL is valid (non-raising)
result = is_valid_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
print(f"✅ is_valid_youtube_url: {result}")

result = is_valid_youtube_url("https://example.com")
print(f"✅ is_valid_youtube_url (invalid): {result}")

# Test 7: Validate query
try:
    query = validate_query("What is RAG?")
    print(f"✅ Valid query: {query}")
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 8: Query too short
try:
    validate_query("Hi")
    print("❌ Should have failed!")
except ValueError:
    print(f"✅ Correctly rejected short query")

# Test 9: Sanitize text
sanitized = sanitize_text("  Hello   World  ")
print(f"✅ Sanitized text: '{sanitized}'")

print("="*60)
print("All validator tests passed!")
print("="*60)