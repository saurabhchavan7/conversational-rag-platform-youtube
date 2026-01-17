import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Test starting...")

try:
    from indexing.document_loader import YouTubeTranscriptLoader
    print("Import successful!")
    
    loader = YouTubeTranscriptLoader()
    print("Loader created!")
    
    result = loader.load("Gfr50f6ZBvo")
    print(f"Transcript loaded! {result['num_segments']} segments")
    print(f"First 100 chars: {result['text'][:100]}")
    
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()

print("Test complete!")