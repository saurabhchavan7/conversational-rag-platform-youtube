"""
Text Splitter for Document Chunking
Splits long texts into smaller chunks for optimal retrieval
"""

from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class TranscriptTextSplitter:
    """
    Split YouTube transcripts into chunks for embedding
    
    Features:
    - Configurable chunk size and overlap
    - Preserves semantic coherence
    - Metadata preservation
    - Character-based splitting
    """
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize text splitter
        
        Args:
            chunk_size: Maximum characters per chunk (default: from settings)
            chunk_overlap: Characters overlap between chunks (default: from settings)
            separators: List of separators for splitting (default: ["\n\n", "\n", " ", ""])
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        
        # Initialize LangChain's RecursiveCharacterTextSplitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False
        )
        
        logger.info(
            f"Initialized TranscriptTextSplitter with "
            f"chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}"
        )
    
    def split_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, any]]:
        """
        Split text into chunks with metadata
        
        Args:
            text: Text to split
            metadata: Optional metadata to attach to each chunk
        
        Returns:
            List of chunk dictionaries
        """
        if not text:
            logger.warning("Empty text provided to splitter")
            return []
        
        # Split text using LangChain splitter
        text_chunks = self.splitter.split_text(text)
        
        logger.info(
            f"Split text into {len(text_chunks)} chunks "
            f"(original length: {len(text)} chars)"
        )
        
        # Create chunk objects with metadata
        chunks = []
        for idx, chunk_text in enumerate(text_chunks):
            chunk = {
                "text": chunk_text,
                "chunk_id": idx,
                "chunk_length": len(chunk_text),
                "metadata": metadata or {}
            }
            chunks.append(chunk)
        
        return chunks
    
    def split_transcript(
        self,
        transcript_data: Dict[str, any]
    ) -> List[Dict[str, any]]:
        """
        Split transcript data from document loader
        
        Args:
            transcript_data: Dictionary from YouTubeTranscriptLoader.load()
        
        Returns:
            List of chunks with full metadata
        """
        # Extract data
        text = transcript_data.get("text", "")
        video_id = transcript_data.get("video_id", "unknown")
        language = transcript_data.get("language", "unknown")
        
        # Create metadata
        metadata = {
            "video_id": video_id,
            "language": language,
            "source": "youtube",
            "total_chars": len(text)
        }
        
        logger.info(f"Splitting transcript for video {video_id}")
        
        # Split text
        chunks = self.split_text(text, metadata)
        
        logger.info(
            f"Created {len(chunks)} chunks for video {video_id} "
            f"(original: {len(text)} chars)"
        )
        
        return chunks
    
    def get_chunk_stats(self, chunks: List[Dict[str, any]]) -> Dict[str, any]:
        """
        Calculate statistics about chunks
        
        Args:
            chunks: List of chunk dictionaries
        
        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_chars": 0,
                "avg_chunk_length": 0,
                "min_chunk_length": 0,
                "max_chunk_length": 0
            }
        
        chunk_lengths = [c["chunk_length"] for c in chunks]
        
        stats = {
            "total_chunks": len(chunks),
            "total_chars": sum(chunk_lengths),
            "avg_chunk_length": sum(chunk_lengths) / len(chunks),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths)
        }
        
        return stats


def split_transcript_into_chunks(
    transcript_data: Dict[str, any],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> List[Dict[str, any]]:
    """
    Convenience function to split transcript
    
    Args:
        transcript_data: Dictionary from document loader
        chunk_size: Optional chunk size override
        chunk_overlap: Optional overlap override
    
    Returns:
        List of chunks
    """
    splitter = TranscriptTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_transcript(transcript_data)