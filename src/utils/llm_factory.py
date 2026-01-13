"""
LLM factory utility for initializing language models.

Supports multiple providers (OpenAI, Azure OpenAI, local, etc.)
and abstracts model initialization behind a unified interface.
"""

import os
from typing import Optional, Union, Literal
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.language_model import LanguageModel


class LLMFactory:
    """
    Factory for creating language model instances.
    
    Supports multiple providers and configurations:
    - OpenAI: GPT-4, GPT-3.5-turbo
    - Azure OpenAI: For corporate deployments
    - Local models: Via Ollama or similar
    """
    
    @staticmethod
    def create_llm(
        provider: str = "openai",
        model: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: int = 30,
        **kwargs
    ) -> LanguageModel:
        """
        Create an LLM instance based on provider and config.
        
        Args:
            provider: "openai" | "azure" | "local"
            model: Model identifier (e.g., "gpt-4o", "gpt-3.5-turbo")
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
            **kwargs: Additional provider-specific arguments
        
        Returns:
            LanguageModel instance
        
        Raises:
            ValueError: If provider not recognized or required env vars missing
        """
        
        if provider == "openai":
            api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            return ChatOpenAI(
                model_name=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                request_timeout=timeout,
                top_p=kwargs.get("top_p", 0.9),
            )
        
        elif provider == "azure":
            api_key = kwargs.get("api_key") or os.getenv("AZURE_OPENAI_API_KEY")
            api_version = kwargs.get("api_version") or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
            azure_endpoint = kwargs.get("azure_endpoint") or os.getenv("AZURE_OPENAI_ENDPOINT")
            deployment_name = kwargs.get("deployment_name") or os.getenv("AZURE_OPENAI_DEPLOYMENT")
            
            if not all([api_key, azure_endpoint, deployment_name]):
                raise ValueError("Azure OpenAI configuration incomplete")
            
            return AzureChatOpenAI(
                model_name=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint,
                azure_deployment=deployment_name,
                request_timeout=timeout,
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    @staticmethod
    def create_supervisor_llm(**kwargs) -> LanguageModel:
        """Create an LLM configured for supervision tasks (lower temperature)."""
        return LLMFactory.create_llm(
            temperature=0.1,
            max_tokens=2048,
            **kwargs
        )
    
    @staticmethod
    def create_worker_llm(worker_type: str, **kwargs) -> LanguageModel:
        """Create an LLM configured for specific worker tasks."""
        if worker_type == "sql":
            return LLMFactory.create_llm(
                temperature=0.0,  # Deterministic for SQL
                max_tokens=2048,
                **kwargs
            )
        elif worker_type == "vector":
            return LLMFactory.create_llm(
                temperature=0.2,  # Slightly more creative for summarization
                max_tokens=2048,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown worker type: {worker_type}")
