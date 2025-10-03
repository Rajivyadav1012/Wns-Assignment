"""
LLM Provider Integration
Step 4: LLM Providers (OpenAI, Anthropic, Google, Groq)
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Generator
import time
from config.settings import settings
from utils.logger import setup_logger
from utils.observability import observability

logger = setup_logger("llm_providers")

# ==================== Base LLM Provider ====================

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.provider_name = self.__class__.__name__.replace("Provider", "")
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False
    ) -> str:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Generator[str, None, None]:
        """Generate streaming response from LLM"""
        pass

# ==================== OpenAI Provider ====================

class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM Provider"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        super().__init__(api_key, model)
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            logger.info(f"✅ OpenAI provider initialized with model: {model}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False
    ) -> str:
        """Generate response from OpenAI"""
        try:
            start_time = time.time()
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            latency = time.time() - start_time
            
            # Log to observability
            observability.log_llm_call(
                provider="OpenAI",
                model=self.model,
                prompt=prompt,
                response=content,
                tokens_used=tokens_used,
                latency=latency
            )
            
            return content
            
        except Exception as e:
            logger.error(f"❌ OpenAI generation failed: {e}")
            raise
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Generator[str, None, None]:
        """Generate streaming response from OpenAI"""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"❌ OpenAI streaming failed: {e}")
            raise

# ==================== Anthropic Provider ====================

class AnthropicProvider(BaseLLMProvider):
    """Anthropic (Claude) LLM Provider"""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        super().__init__(api_key, model)
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
            logger.info(f"✅ Anthropic provider initialized with model: {model}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Anthropic: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False
    ) -> str:
        """Generate response from Anthropic"""
        try:
            start_time = time.time()
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            latency = time.time() - start_time
            
            observability.log_llm_call(
                provider="Anthropic",
                model=self.model,
                prompt=prompt,
                response=content,
                tokens_used=tokens_used,
                latency=latency
            )
            
            return content
            
        except Exception as e:
            logger.error(f"❌ Anthropic generation failed: {e}")
            raise
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Generator[str, None, None]:
        """Generate streaming response from Anthropic"""
        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            logger.error(f"❌ Anthropic streaming failed: {e}")
            raise

# ==================== Google (Gemini) Provider ====================

class GoogleProvider(BaseLLMProvider):
    """Google Gemini LLM Provider"""
    
    def __init__(self, api_key: str, model: str = "gemini-pro"):
        super().__init__(api_key, model)
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
            logger.info(f"✅ Google provider initialized with model: {model}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Google: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False
    ) -> str:
        """Generate response from Google Gemini"""
        try:
            start_time = time.time()
            
            # Combine system prompt with user prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            
            response = self.client.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            content = response.text
            latency = time.time() - start_time
            
            observability.log_llm_call(
                provider="Google",
                model=self.model,
                prompt=prompt,
                response=content,
                tokens_used=None,  # Gemini doesn't provide token count directly
                latency=latency
            )
            
            return content
            
        except Exception as e:
            logger.error(f"❌ Google generation failed: {e}")
            raise
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Generator[str, None, None]:
        """Generate streaming response from Google Gemini"""
        try:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            
            response = self.client.generate_content(
                full_prompt,
                generation_config=generation_config,
                stream=True
            )
            
            for chunk in response:
                yield chunk.text
                    
        except Exception as e:
            logger.error(f"❌ Google streaming failed: {e}")
            raise

# ==================== Groq Provider ====================

class GroqProvider(BaseLLMProvider):
    """Groq LLM Provider (Fast inference with Llama, Mixtral, etc.)"""
    
    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        super().__init__(api_key, model)
        try:
            from groq import Groq
            self.client = Groq(api_key=api_key)
            logger.info(f"✅ Groq provider initialized with model: {model}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Groq: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False
    ) -> str:
        """Generate response from Groq"""
        try:
            start_time = time.time()
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            latency = time.time() - start_time
            
            # Log to observability
            observability.log_llm_call(
                provider="Groq",
                model=self.model,
                prompt=prompt,
                response=content,
                tokens_used=tokens_used,
                latency=latency
            )
            
            logger.info(f"🚀 Groq response: {tokens_used} tokens in {latency:.2f}s")
            
            return content
            
        except Exception as e:
            logger.error(f"❌ Groq generation failed: {e}")
            raise
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Generator[str, None, None]:
        """Generate streaming response from Groq"""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"❌ Groq streaming failed: {e}")
            raise

# ==================== LLM Factory ====================

class LLMFactory:
    """Factory class to create LLM provider instances"""
    
    @staticmethod
    def create_provider(
        provider_name: str,
        model: Optional[str] = None
    ) -> BaseLLMProvider:
        """
        Create an LLM provider instance
        
        Args:
            provider_name: Name of the provider (OpenAI, Anthropic, Google, Groq)
            model: Optional model name (uses default if not provided)
            
        Returns:
            LLM provider instance
        """
        provider_name = provider_name.lower()
        
        if provider_name == "openai":
            api_key = settings.OPENAI_API_KEY
            default_model = "gpt-3.5-turbo"
            return OpenAIProvider(api_key, model or default_model)
            
        elif provider_name == "anthropic":
            api_key = settings.ANTHROPIC_API_KEY
            default_model = "claude-3-sonnet-20240229"
            return AnthropicProvider(api_key, model or default_model)
            
        elif provider_name == "google":
            api_key = settings.GOOGLE_API_KEY
            default_model = "gemini-pro"
            return GoogleProvider(api_key, model or default_model)
            
        elif provider_name == "groq":
            api_key = settings.GROQ_API_KEY
            default_model = "llama-3.1-8b-instant"
            return GroqProvider(api_key, model or default_model)
            
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
    
    @staticmethod
    def get_available_providers() -> List[str]:
        """Get list of available providers based on configured API keys"""
        return settings.get_available_providers()

# ==================== Test Function ====================

def test_llm_providers():
    """Test LLM providers"""
    print("\n" + "="*70)
    print("🤖 TESTING LLM PROVIDERS")
    print("="*70 + "\n")
    
    # Get available providers
    available = LLMFactory.get_available_providers()
    print(f"📋 Available providers: {', '.join(available)}\n")
    
    if not available:
        print("❌ No providers configured! Please add API keys to .env file")
        return
    
    # Test each available provider
    for provider_name in available:
        print(f"\n{'='*70}")
        print(f"Testing {provider_name}...")
        print('='*70)
        
        try:
            # Create provider
            provider = LLMFactory.create_provider(provider_name)
            
            # Test generation
            prompt = "Say 'Hello World' and explain what it means in one sentence."
            system_prompt = "You are a helpful assistant."
            
            print(f"\n📝 Prompt: {prompt}")
            print(f"⏳ Generating response...")
            
            response = provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=100
            )
            
            print(f"\n✅ Response from {provider_name}:")
            print(f"   {response}\n")
            
            # Test streaming
            print(f"🌊 Testing streaming...")
            print(f"   Response: ", end="")
            for chunk in provider.generate_stream(
                prompt="Count from 1 to 5.",
                max_tokens=50
            ):
                print(chunk, end="", flush=True)
            print("\n")
            
        except Exception as e:
            print(f"❌ {provider_name} test failed: {e}\n")
    
    print("="*70)
    print("✅ LLM provider tests complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_llm_providers()