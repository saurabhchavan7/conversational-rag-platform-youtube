"""
Document Loader for YouTube Transcripts
Fetches and processes YouTube video transcripts
"""

from typing import Dict, List, Optional
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable
)

from config.logging_config import get_logger
from utils.exceptions import (
    TranscriptFetchError,
    TranscriptNotAvailableError,
    InvalidVideoIDError
)
from utils.validators import validate_youtube_video_id

logger = get_logger(__name__)


class YouTubeTranscriptLoader:
    """
    Load transcripts from YouTube videos
    
    Features:
    - Automatic language detection
    - Fallback to available languages
    - Transcript concatenation
    - Metadata extraction
    """
    
    def __init__(self, preferred_languages: Optional[List[str]] = None):
        """
        Initialize transcript loader
        
        Args:
            preferred_languages: List of preferred language codes (default: ["en"])
        """
        self.preferred_languages = preferred_languages or ["en"]
        logger.info(f"Initialized YouTubeTranscriptLoader with languages: {self.preferred_languages}")
    
    def load(self, video_id: str) -> Dict[str, any]:
        """
        Load transcript for a YouTube video
        
        Args:
            video_id: YouTube video ID (11 characters)
        
        Returns:
            Dictionary containing:
                - text: Full transcript text
                - video_id: Video ID
                - language: Transcript language
                - chunks: List of transcript segments with timestamps
        
        Raises:
            InvalidVideoIDError: If video ID format is invalid
            TranscriptNotAvailableError: If transcript is disabled
            TranscriptFetchError: If fetching fails
        
        Example:
            >>> loader = YouTubeTranscriptLoader()
            >>> result = loader.load("dQw4w9WgXcQ")
            >>> print(result["text"][:100])
        """
        # Validate video ID
        video_id = validate_youtube_video_id(video_id)
        
        logger.info(f"Fetching transcript for video: {video_id}")
        
        try:
            # Attempt to fetch transcript
            transcript_list = YouTubeTranscriptApi.get_transcript(
                video_id,
                languages=self.preferred_languages
            )
            
            # Extract language info
            language = self._detect_language(transcript_list)
            
            # Concatenate transcript text
            full_text = self._concatenate_transcript(transcript_list)
            
            # Prepare result
            result = {
                "text": full_text,
                "video_id": video_id,
                "language": language,
                "chunks": transcript_list,  # Keep original segments with timestamps
                "num_segments": len(transcript_list),
                "total_chars": len(full_text)
            }
            
            logger.info(
                f"Successfully loaded transcript: {len(transcript_list)} segments, "
                f"{len(full_text)} characters"
            )
            
            return result
        
        except TranscriptsDisabled:
            error_msg = f"Transcripts are disabled for video: {video_id}"
            logger.error(error_msg)
            raise TranscriptNotAvailableError(error_msg)
        
        except NoTranscriptFound:
            error_msg = (
                f"No transcript found for video: {video_id} "
                f"in languages: {self.preferred_languages}"
            )
            logger.error(error_msg)
            raise TranscriptNotAvailableError(error_msg)
        
        except VideoUnavailable:
            error_msg = f"Video unavailable: {video_id}"
            logger.error(error_msg)
            raise TranscriptFetchError(error_msg)
        
        except Exception as e:
            error_msg = f"Failed to fetch transcript for {video_id}: {str(e)}"
            logger.error(error_msg)
            raise TranscriptFetchError(error_msg)
    
    def _concatenate_transcript(self, transcript_list: List[Dict]) -> str:
        """
        Concatenate transcript segments into full text
        
        Args:
            transcript_list: List of transcript segments
        
        Returns:
            Concatenated text with proper spacing
        """
        # Join all text segments with spaces
        full_text = " ".join(segment["text"] for segment in transcript_list)
        
        # Clean up extra whitespace
        full_text = " ".join(full_text.split())
        
        return full_text
    
    def _detect_language(self, transcript_list: List[Dict]) -> str:
        """
        Detect transcript language
        
        Args:
            transcript_list: List of transcript segments
        
        Returns:
            Language code (e.g., "en", "es")
        """
        # For now, assume first preferred language
        # In production, could detect from transcript metadata
        return self.preferred_languages[0]
    
    def list_available_transcripts(self, video_id: str) -> List[Dict[str, str]]:
        """
        List all available transcripts for a video
        
        Args:
            video_id: YouTube video ID
        
        Returns:
            List of available transcripts with language info
        
        Example:
            >>> loader = YouTubeTranscriptLoader()
            >>> transcripts = loader.list_available_transcripts("dQw4w9WgXcQ")
            >>> print(transcripts)
            [{"language": "en", "language_code": "en"}]
        """
        video_id = validate_youtube_video_id(video_id)
        
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            available = []
            for transcript in transcript_list:
                available.append({
                    "language": transcript.language,
                    "language_code": transcript.language_code,
                    "is_generated": transcript.is_generated,
                    "is_translatable": transcript.is_translatable
                })
            
            logger.info(f"Found {len(available)} transcripts for video {video_id}")
            return available
        
        except Exception as e:
            logger.error(f"Failed to list transcripts for {video_id}: {e}")
            raise TranscriptFetchError(f"Failed to list transcripts: {e}")


def load_youtube_transcript(
    video_id: str,
    languages: Optional[List[str]] = None
) -> Dict[str, any]:
    """
    Convenience function to load transcript
    
    Args:
        video_id: YouTube video ID
        languages: Preferred languages (default: ["en"])
    
    Returns:
        Transcript data dictionary
    
    Example:
        >>> transcript = load_youtube_transcript("dQw4w9WgXcQ")
        >>> print(transcript["text"][:100])
    """
    loader = YouTubeTranscriptLoader(preferred_languages=languages)
    return loader.load(video_id)