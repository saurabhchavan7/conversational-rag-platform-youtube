"""
LLM Client for Answer Generation
Handles interactions with OpenAI API for generating answers
"""

from typing import Dict, List, Optional
from openai import OpenAI

from config.logging_config import get_logger
from config.settings import settings
from utils.exceptions import LLMError, PromptTooLongError

logger = get_logger(__name__)


class LLMClient:
    """
    Client for LLM-based answer generation
    
    Features:
    - OpenAI API integration
    - Streaming support
    - Token limit validation
    - Error handling and retries
    - Response parsing
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 500
    ):
        """
        Initialize LLM client
        
        Args:
            model: LLM model name (default: from settings)
            temperature: Sampling temperature (default: from settings)
            max_tokens: Maximum tokens in response
        """
        self.model = model or settings.OPENAI_CHAT_MODEL
        self.temperature = temperature or settings.OPENAI_TEMPERATURE
        self.max_tokens = max_tokens
        
        # Initialize OpenAI client
        try:
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info(
                f"Initialized LLMClient with model={self.model}, "
                f"temperature={self.temperature}, max_tokens={self.max_tokens}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise LLMError(f"LLM client initialization failed: {e}")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate response from LLM
        
        Args:
            prompt: Complete prompt string
            temperature: Override temperature
            max_tokens: Override max tokens
        
        Returns:
            Generated text response
        
        Raises:
            LLMError: If generation fails
            PromptTooLongError: If prompt exceeds token limits
        
        Example:
            >>> client = LLMClient()
            >>> answer = client.generate("What is RAG?\n\nContext: RAG is...")
            >>> print(answer)
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        logger.info(f"Generating response for prompt ({len(prompt)} chars)")
        
        # Check prompt length (rough estimate)
        estimated_tokens = len(prompt) // 4
        if estimated_tokens > 3500:  # Leave room for response
            raise PromptTooLongError(
                f"Prompt too long: ~{estimated_tokens} tokens. "
                "Consider reducing context or chunk count."
            )
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temp,
                max_tokens=max_tok
            )
            
            # Extract generated text
            answer = response.choices[0].message.content.strip()
            
            # Log token usage
            if hasattr(response, 'usage'):
                logger.info(
                    f"Token usage - Prompt: {response.usage.prompt_tokens}, "
                    f"Completion: {response.usage.completion_tokens}, "
                    f"Total: {response.usage.total_tokens}"
                )
            
            logger.info(f"Generated answer: {len(answer)} characters")
            
            return answer
        
        except Exception as e:
            error_msg = f"LLM generation failed: {str(e)}"
            logger.error(error_msg)
            raise LLMError(error_msg)
    
    def generate_with_system_prompt(
        self,
        user_message: str,
        system_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate with separate system and user messages
        
        Args:
            user_message: User's message
            system_prompt: System instructions
            temperature: Override temperature
            max_tokens: Override max tokens
        
        Returns:
            Generated response
        
        Example:
            >>> client = LLMClient()
            >>> answer = client.generate_with_system_prompt(
            ...     user_message="What is RAG?",
            ...     system_prompt="You are a helpful assistant..."
            ... )
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        logger.info("Generating with system prompt")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=temp,
                max_tokens=max_tok
            )
            
            answer = response.choices[0].message.content.strip()
            
            logger.info(f"Generated answer: {len(answer)} characters")
            
            return answer
        
        except Exception as e:
            error_msg = f"LLM generation failed: {str(e)}"
            logger.error(error_msg)
            raise LLMError(error_msg)
    
    def generate_streaming(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Generate response with streaming (yields chunks)
        
        Args:
            prompt: Complete prompt
            temperature: Override temperature
            max_tokens: Override max tokens
        
        Yields:
            Text chunks as they're generated
        
        Example:
            >>> client = LLMClient()
            >>> for chunk in client.generate_streaming("What is AI?"):
            >>>     print(chunk, end="", flush=True)
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        logger.info("Starting streaming generation")
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temp,
                max_tokens=max_tok,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        
        except Exception as e:
            error_msg = f"Streaming generation failed: {str(e)}"
            logger.error(error_msg)
            raise LLMError(error_msg)


def generate_answer(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 500
) -> str:
    """
    Convenience function to generate answer
    
    Args:
        prompt: Complete prompt with context and question
        model: LLM model to use
        temperature: Sampling temperature
        max_tokens: Max response tokens
    
    Returns:
        Generated answer
    
    Example:
        >>> from augmentation.prompt_templates import build_qa_prompt
        >>> prompt = build_qa_prompt(question, chunks)
        >>> answer = generate_answer(prompt)
        >>> print(answer)
    """
    client = LLMClient(model=model, temperature=temperature, max_tokens=max_tokens)
    return client.generate(prompt)