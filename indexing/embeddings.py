"""
Embedding Generator for Text Chunks
Generates vector embeddings using OpenAI API
"""

from typing import List, Dict, Optional
from openai import OpenAI

from config.logging_config import get_logger
from config.settings import settings
from utils.exceptions import EmbeddingGenerationError

logger = get_logger(__name__)


class EmbeddingGenerator:
    """
    Generate embeddings for text chunks using OpenAI
    
    Features:
    - Configurable embedding model
    - Batch processing support
    - Dimension control (512D or 1536D)
    - Error handling and retries
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        dimensions: Optional[int] = None
    ):
        """
        Initialize embedding generator
        
        Args:
            model: OpenAI embedding model (default: from settings)
            dimensions: Embedding dimensions (default: 512 from settings)
        """
        self.model = model or settings.OPENAI_EMBEDDING_MODEL
        self.dimensions = dimensions or settings.EMBEDDING_DIMENSIONS
        
        # Initialize OpenAI client
        try:
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info(
                f"Initialized EmbeddingGenerator with model={self.model}, "
                f"dimensions={self.dimensions}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise EmbeddingGenerationError(f"OpenAI initialization failed: {e}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector (list of floats)
        
        Raises:
            EmbeddingGenerationError: If embedding generation fails
        
        Example:
            >>> generator = EmbeddingGenerator()
            >>> embedding = generator.generate_embedding("Hello world")
            >>> len(embedding)
            512
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            raise EmbeddingGenerationError("Cannot generate embedding for empty text")
        
        try:
            # Call OpenAI API
            response = self.client.embeddings.create(
                input=text,
                model=self.model,
                dimensions=self.dimensions
            )
            
            # Extract embedding vector
            embedding = response.data[0].embedding
            
            logger.debug(f"Generated embedding with {len(embedding)} dimensions")
            
            return embedding
        
        except Exception as e:
            error_msg = f"Failed to generate embedding: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingGenerationError(error_msg)
    
    def generate_embeddings_batch(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to log progress
        
        Returns:
            List of embedding vectors
        
        Example:
            >>> generator = EmbeddingGenerator()
            >>> texts = ["Hello", "World", "Test"]
            >>> embeddings = generator.generate_embeddings_batch(texts)
            >>> len(embeddings)
            3
        """
        if not texts:
            logger.warning("Empty text list provided for batch embedding")
            return []
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        embeddings = []
        for idx, text in enumerate(texts):
            try:
                embedding = self.generate_embedding(text)
                embeddings.append(embedding)
                
                if show_progress and (idx + 1) % 10 == 0:
                    logger.info(f"Progress: {idx + 1}/{len(texts)} embeddings generated")
            
            except EmbeddingGenerationError as e:
                logger.error(f"Failed to generate embedding for text {idx}: {e}")
                # Continue with next text instead of failing entire batch
                embeddings.append(None)
        
        logger.info(f"Successfully generated {len([e for e in embeddings if e])} embeddings")
        
        return embeddings
    
    def embed_chunks(
        self,
        chunks: List[Dict[str, any]],
        show_progress: bool = True
    ) -> List[Dict[str, any]]:
        """
        Generate embeddings for text chunks
        
        Args:
            chunks: List of chunk dictionaries from text splitter
            show_progress: Whether to log progress
        
        Returns:
            List of chunks with embeddings added
        
        Example:
            >>> from indexing.text_splitter import split_transcript_into_chunks
            >>> from indexing.document_loader import load_youtube_transcript
            >>> 
            >>> transcript = load_youtube_transcript("abc123")
            >>> chunks = split_transcript_into_chunks(transcript)
            >>> 
            >>> generator = EmbeddingGenerator()
            >>> embedded_chunks = generator.embed_chunks(chunks)
        """
        if not chunks:
            logger.warning("Empty chunks list provided")
            return []
        
        logger.info(f"Embedding {len(chunks)} chunks")
        
        # Extract texts from chunks
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.generate_embeddings_batch(texts, show_progress=show_progress)
        
        # Add embeddings to chunks
        embedded_chunks = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if embedding is not None:
                chunk_with_embedding = chunk.copy()
                chunk_with_embedding["embedding"] = embedding
                chunk_with_embedding["embedding_dim"] = len(embedding)
                embedded_chunks.append(chunk_with_embedding)
            else:
                logger.warning(f"Skipping chunk {idx} due to embedding failure")
        
        logger.info(
            f"Successfully embedded {len(embedded_chunks)}/{len(chunks)} chunks"
        )
        
        return embedded_chunks


def generate_embeddings_for_chunks(
    chunks: List[Dict[str, any]],
    model: Optional[str] = None,
    dimensions: Optional[int] = None
) -> List[Dict[str, any]]:
    """
    Convenience function to generate embeddings
    
    Args:
        chunks: List of chunk dictionaries
        model: Optional model override
        dimensions: Optional dimensions override
    
    Returns:
        Chunks with embeddings
    
    Example:
        >>> from indexing.text_splitter import split_transcript_into_chunks
        >>> from indexing.document_loader import load_youtube_transcript
        >>> 
        >>> transcript = load_youtube_transcript("abc123")
        >>> chunks = split_transcript_into_chunks(transcript)
        >>> embedded_chunks = generate_embeddings_for_chunks(chunks)
    """
    generator = EmbeddingGenerator(model=model, dimensions=dimensions)
    return generator.embed_chunks(chunks)