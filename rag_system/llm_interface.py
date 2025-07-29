# rag_system/llm_interface.py
"""
LLM interface module for the RAG system.
Handles language model integration for answer generation.
"""

import logging
from typing import Optional, Dict, Any
import re

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class LLMInterface:
    """
    Interface for language model integration in the RAG system.
    Supports various open-source models through HuggingFace transformers.
    """
    
    # Recommended models with their characteristics
    RECOMMENDED_MODELS = {
        'microsoft/DialoGPT-medium': {'type': 'conversational', 'size': 'medium'},
        'microsoft/DialoGPT-small': {'type': 'conversational', 'size': 'small'},
        'distilgpt2': {'type': 'generative', 'size': 'small'},
        'gpt2': {'type': 'generative', 'size': 'medium'},
        'facebook/blenderbot-400M-distill': {'type': 'conversational', 'size': 'medium'},
    }
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium",
                 max_tokens: int = 512, temperature: float = 0.7,
                 device: Optional[str] = None):
        """
        Initialize the LLM interface.
        
        Args:
            model_name: Name of the HuggingFace model to use
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature for generation
            device: Device to run the model on ('cpu', 'cuda', etc.)
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("transformers not available. Using fallback text generation.")
            self.model = None
            self.tokenizer = None
            self.pipeline = None
        else:
            self.model_name = model_name
            self.max_tokens = max_tokens
            self.temperature = temperature
            self.device = device or ("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
            
            self._load_model()
    
    def _load_model(self):
        """Load the language model and tokenizer."""
        try:
            logger.info(f"Loading LLM model: {self.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" and TORCH_AVAILABLE else None
            )
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            logger.warning("Falling back to rule-based text generation")
            self.pipeline = None
    
    def generate_answer(self, question: str, context: str, 
                       custom_prompt: Optional[str] = None) -> str:
        """
        Generate an answer based on the question and retrieved context.
        
        Args:
            question: User's question
            context: Retrieved context from vector search
            custom_prompt: Custom prompt template (optional)
            
        Returns:
            Generated answer string
        """
        if self.pipeline is None:
            return self._fallback_answer_generation(question, context)
        
        try:
            # Prepare the prompt
            if custom_prompt:
                prompt = custom_prompt.format(context=context, question=question)
            else:
                prompt = self._create_default_prompt(question, context)
            
            # Generate response
            response = self.pipeline(
                prompt,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            # Extract generated text
            generated_text = response[0]['generated_text'].strip()
            
            # Clean up the response
            answer = self._clean_generated_text(generated_text, question)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return self._fallback_answer_generation(question, context)
    
    def _create_default_prompt(self, question: str, context: str) -> str:
        """Create a default prompt template for answer generation."""
        prompt = f"""Context: {context}

Question: {question}

Based on the provided context, please answer the question clearly and concisely. If the context doesn't contain enough information to answer the question, say so.

Answer:"""
        
        return prompt
    
    def _clean_generated_text(self, text: str, question: str) -> str:
        """Clean and post-process generated text."""
        # Remove repetitive patterns
        text = re.sub(r'(.+?)\1{2,}', r'\1', text)
        
        # Remove incomplete sentences at the end
        sentences = text.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            text = '.'.join(sentences[:-1]) + '.'
        
        # Limit length
        words = text.split()
        if len(words) > 200:  # Reasonable limit
            text = ' '.join(words[:200]) + '...'
        
        # Ensure the answer doesn't just repeat the question
        if text.lower().startswith(question.lower()):
            text = text[len(question):].strip()
        
        return text.strip()
    
    def _fallback_answer_generation(self, question: str, context: str) -> str:
        """
        Fallback method for answer generation when no LLM is available.
        Uses simple extractive approach.
        """
        logger.info("Using fallback answer generation")
        
        # Simple extractive approach
        sentences = context.split('.')
        question_words = set(question.lower().split())
        
        # Find sentences with highest overlap with question
        best_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
                
            sentence_words = set(sentence.lower().split())
            overlap = len(question_words.intersection(sentence_words))
            
            if overlap > 0:
                best_sentences.append((sentence, overlap))
        
        # Sort by overlap and take top sentences
        best_sentences.sort(key=lambda x: x[1], reverse=True)
        
        if best_sentences:
            # Combine top 2-3 sentences
            answer_parts = [sent[0] for sent in best_sentences[:3]]
            answer = '. '.join(answer_parts)
            
            # Add prefix to indicate this is extracted information
            return f"Based on the available information: {answer}"
        else:
            return "I couldn't find specific information to answer your question in the provided context."
    
    def generate_summary(self, text: str, max_length: int = 150) -> str:
        """
        Generate a summary of the given text.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Generated summary
        """
        if self.pipeline is None:
            return self._fallback_summary(text, max_length)
        
        try:
            prompt = f"Summarize the following text in a few sentences:\n\n{text}\n\nSummary:"
            
            response = self.pipeline(
                prompt,
                max_new_tokens=max_length,
                temperature=0.5,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                return_full_text=False
            )
            
            summary = response[0]['generated_text'].strip()
            return self._clean_generated_text(summary, "")
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return self._fallback_summary(text, max_length)
    
    def _fallback_summary(self, text: str, max_length: int) -> str:
        """Fallback summary generation using extractive method."""
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not sentences:
            return "No content to summarize."
        
        # Take first few sentences that fit within max_length
        summary_parts = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) > max_length:
                break
            summary_parts.append(sentence)
            current_length += len(sentence)
        
        return '. '.join(summary_parts) + '.' if summary_parts else sentences[0][:max_length] + '...'
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if self.pipeline is None:
            return {
                'model_name': 'fallback',
                'status': 'using rule-based generation',
                'device': 'cpu'
            }
        
        return {
            'model_name': self.model_name,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'device': self.device,
            'status': 'loaded'
        }
    
    @classmethod
    def list_recommended_models(cls) -> Dict[str, Dict[str, str]]:
        """List recommended models for different use cases."""
        return cls.RECOMMENDED_MODELS.copy()
    
    def set_generation_params(self, max_tokens: Optional[int] = None,
                            temperature: Optional[float] = None):
        """Update generation parameters."""
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if temperature is not None:
            self.temperature = temperature
        
        logger.info(f"Updated generation params: max_tokens={self.max_tokens}, temperature={self.temperature}")