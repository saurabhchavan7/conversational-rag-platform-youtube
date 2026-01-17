"""
Document Loader for YouTube Transcripts
TEMPORARY MOCK VERSION - Fetches sample data for testing
TODO: Fix youtube-transcript-api integration later
"""

from typing import Dict, List, Optional

from config.logging_config import get_logger
from utils.exceptions import (
    TranscriptFetchError,
    TranscriptNotAvailableError,
    InvalidVideoIDError
)
from utils.validators import validate_youtube_video_id

logger = get_logger(__name__)

# Sample transcript for testing (from 3Blue1Brown LLM video)
SAMPLE_TRANSCRIPT = """
Imagine you happen across a short movie script that describes a scene between a person and their AI assistant. The script has what the person asks the AI, but the AI's response has been torn off. Suppose you also have this powerful magical machine that can take any text and provide a sensible prediction of what word comes next. You could then finish the script by feeding in what you have to the machine, seeing what it would predict to start the AI's answer, and then repeating this over and over with a growing script completing the dialogue.

When you interact with a chatbot, this is exactly what's happening. A large language model is a sophisticated mathematical function that predicts what word comes next for any piece of text. Instead of predicting one word with certainty, though, what it does is assign a probability to all possible next words.

To build a chatbot, you lay out some text that describes an interaction between a user and a hypothetical AI assistant, add on whatever the user types in as the first part of the interaction, and then have the model repeatedly predict the next word that such a hypothetical AI assistant would say in response, and that's what's presented to the user.

Models learn how to make these predictions by processing an enormous amount of text, typically pulled from the internet. For a standard human to read the amount of text that was used to train GPT-3, for example, if they read non-stop 24-7, it would take over 2600 years.

You can think of training a little bit like tuning the dials on a big machine. The way that a language model behaves is entirely determined by these many different continuous values, usually called parameters or weights. Changing those parameters will change the probabilities that the model gives for the next word on a given input.

What puts the large in large language model is how they can have hundreds of billions of these parameters. No human ever deliberately sets those parameters. Instead, they begin at random, meaning the model just outputs gibberish, but they're repeatedly refined based on many example pieces of text.
"""


class YouTubeTranscriptLoader:
    """
    MOCK VERSION: Returns sample transcript data
    TODO: Integrate real youtube-transcript-api later
    
    For now, this returns a fixed sample transcript to allow
    development of downstream components (text splitting, embeddings, etc.)
    """
    
    def __init__(self, preferred_languages: Optional[List[str]] = None):
        """
        Initialize transcript loader
        
        Args:
            preferred_languages: List of preferred language codes (default: ["en"])
        """
        self.preferred_languages = preferred_languages or ["en"]
        logger.warning("Using MOCK YouTubeTranscriptLoader - returns sample data only!")
        logger.info(f"Initialized YouTubeTranscriptLoader with languages: {self.preferred_languages}")
    
    def load(self, video_id: str) -> Dict[str, any]:
        """
        Load transcript for a YouTube video (MOCK VERSION)
        
        Args:
            video_id: YouTube video ID (11 characters)
        
        Returns:
            Dictionary containing:
                - text: Full transcript text (sample data)
                - video_id: Video ID
                - language: Transcript language
                - chunks: Empty list (mock)
        
        Raises:
            InvalidVideoIDError: If video ID format is invalid
        """
        # Validate video ID
        video_id = validate_youtube_video_id(video_id)
        
        logger.info(f"Fetching transcript for video: {video_id} (MOCK)")
        logger.warning(f"Returning sample transcript data - not real video transcript!")
        
        # Return mock data
        result = {
            "text": SAMPLE_TRANSCRIPT,
            "video_id": video_id,
            "language": "en",
            "chunks": [],  # Empty for mock
            "num_segments": 6,  # Mock value
            "total_chars": len(SAMPLE_TRANSCRIPT)
        }
        
        logger.info(
            f"Successfully loaded transcript (MOCK): {result['num_segments']} segments, "
            f"{len(SAMPLE_TRANSCRIPT)} characters"
        )
        
        return result
    
    def list_available_transcripts(self, video_id: str) -> List[Dict[str, str]]:
        """
        List all available transcripts for a video (MOCK VERSION)
        
        Args:
            video_id: YouTube video ID
        
        Returns:
            List with single English transcript (mock)
        """
        video_id = validate_youtube_video_id(video_id)
        
        logger.warning("list_available_transcripts: returning mock data")
        
        return [{
            "language": "English",
            "language_code": "en",
            "is_generated": False,
            "is_translatable": False
        }]


def load_youtube_transcript(
    video_id: str,
    languages: Optional[List[str]] = None
) -> Dict[str, any]:
    """
    Convenience function to load transcript (MOCK VERSION)
    
    Args:
        video_id: YouTube video ID
        languages: Preferred languages (default: ["en"])
    
    Returns:
        Transcript data dictionary (sample data)
    """
    loader = YouTubeTranscriptLoader(preferred_languages=languages)
    return loader.load(video_id)