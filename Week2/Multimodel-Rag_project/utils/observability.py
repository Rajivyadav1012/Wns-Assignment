"""
Observability and tracing using Langsmith
Step 2: Logging & Observability
"""
import os
from functools import wraps
from typing import Any, Callable, Dict, Optional
import time
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger("observability")

class ObservabilityManager:
    """Manage observability and tracing with Langsmith"""
    
    def __init__(self):
        self.enabled = settings.ENABLE_TRACING and bool(settings.LANGSMITH_API_KEY)
        self.project_name = settings.LANGSMITH_PROJECT
        self.traces = []  # Store traces in memory for non-Langsmith mode
        
        if self.enabled:
            self._initialize_langsmith()
        else:
            logger.warning("⚠️  Observability disabled - Langsmith API key not configured")
            logger.info("📊 Running in local tracing mode")
    
    def _initialize_langsmith(self):
        """Initialize Langsmith tracing"""
        try:
            # Set environment variables for Langsmith
            os.environ["LANGCHAIN_TRACING_V2"] = settings.LANGCHAIN_TRACING_V2
            os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
            os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
            os.environ["LANGCHAIN_PROJECT"] = self.project_name
            
            logger.info(f"✅ Langsmith tracing initialized for project: {self.project_name}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Langsmith: {e}")
            self.enabled = False
    
    def trace_function(
        self,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Decorator to trace function execution
        
        Args:
            name: Custom name for the trace
            metadata: Additional metadata to log
            
        Usage:
            @observability.trace_function(name="my_function")
            def my_function():
                pass
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                trace_name = name or func.__name__
                start_time = time.time()
                
                try:
                    # Log function start
                    logger.debug(f"🔹 Starting: {trace_name}")
                    
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Calculate execution time
                    execution_time = time.time() - start_time
                    
                    # Log function completion
                    logger.debug(f"✅ Completed: {trace_name} ({execution_time:.2f}s)")
                    
                    # Store trace
                    self._store_trace(trace_name, "success", execution_time, metadata)
                    
                    return result
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    logger.error(f"❌ Error in {trace_name}: {str(e)} ({execution_time:.2f}s)")
                    self._store_trace(trace_name, "error", execution_time, metadata, str(e))
                    raise
            
            return wrapper
        return decorator
    
    def _store_trace(
        self,
        name: str,
        status: str,
        duration: float,
        metadata: Optional[Dict] = None,
        error: Optional[str] = None
    ):
        """Store trace information"""
        trace = {
            "name": name,
            "status": status,
            "duration": duration,
            "metadata": metadata or {},
            "error": error,
            "timestamp": time.time()
        }
        self.traces.append(trace)
    
    def log_event(
        self,
        event_name: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        """
        Log a custom event
        
        Args:
            event_name: Name of the event
            properties: Event properties
        """
        try:
            logger.info(f"📊 Event: {event_name}", extra={"properties": properties or {}})
            self._store_trace(event_name, "event", 0, properties)
        except Exception as e:
            logger.error(f"Failed to log event {event_name}: {e}")
    
    def log_llm_call(
        self,
        provider: str,
        model: str,
        prompt: str,
        response: str,
        tokens_used: Optional[int] = None,
        latency: Optional[float] = None
    ):
        """
        Log LLM API call
        
        Args:
            provider: LLM provider name (OpenAI, Anthropic, etc.)
            model: Model name
            prompt: Input prompt
            response: Model response
            tokens_used: Number of tokens used
            latency: Response time in seconds
        """
        properties = {
            "provider": provider,
            "model": model,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "tokens_used": tokens_used,
            "latency_seconds": latency
        }
        
        logger.info(f"🤖 LLM Call: {provider}/{model} - {tokens_used} tokens in {latency:.2f}s")
        self.log_event("llm_call", properties)
    
    def log_retrieval(
        self,
        query: str,
        results_count: int,
        rag_type: str,
        retrieval_time: Optional[float] = None
    ):
        """
        Log retrieval operation
        
        Args:
            query: Search query
            results_count: Number of results retrieved
            rag_type: Type of RAG used (Standard/KG/Hybrid)
            retrieval_time: Time taken for retrieval
        """
        properties = {
            "query_length": len(query),
            "results_count": results_count,
            "rag_type": rag_type,
            "retrieval_time_seconds": retrieval_time
        }
        
        logger.info(f"🔍 Retrieval: {rag_type} - {results_count} results in {retrieval_time:.2f}s")
        self.log_event("retrieval", properties)
    
    def log_guardrails_check(
        self,
        content: str,
        is_toxic: bool,
        is_nsfw: bool,
        toxicity_score: Optional[float] = None
    ):
        """
        Log guardrails check
        
        Args:
            content: Content that was checked
            is_toxic: Whether content is toxic
            is_nsfw: Whether content is NSFW
            toxicity_score: Toxicity score
        """
        properties = {
            "content_length": len(content),
            "is_toxic": is_toxic,
            "is_nsfw": is_nsfw,
            "toxicity_score": toxicity_score
        }
        
        if is_toxic or is_nsfw:
            logger.warning(f"🛡️  Guardrails: Content flagged (toxic={is_toxic}, nsfw={is_nsfw})")
        else:
            logger.info(f"🛡️  Guardrails: Content passed")
        
        self.log_event("guardrails_check", properties)
    
    def log_document_processing(
        self,
        filename: str,
        file_type: str,
        chunks_created: int,
        processing_time: float
    ):
        """Log document processing"""
        properties = {
            "filename": filename,
            "file_type": file_type,
            "chunks_created": chunks_created,
            "processing_time": processing_time
        }
        
        logger.info(f"📄 Document: {filename} - {chunks_created} chunks in {processing_time:.2f}s")
        self.log_event("document_processing", properties)
    
    def get_traces(self, limit: int = 10) -> list:
        """Get recent traces"""
        return self.traces[-limit:]
    
    def clear_traces(self):
        """Clear all traces"""
        self.traces = []
        logger.info("🗑️  Traces cleared")

# Global observability manager instance
observability = ObservabilityManager()

# Test function
def test_observability():
    """Test observability functionality"""
    
    @observability.trace_function(name="test_function")
    def sample_function():
        time.sleep(0.1)
        return "Success"
    
    # Test function tracing
    result = sample_function()
    
    # Test event logging
    observability.log_event("test_event", {"key": "value"})
    
    # Test LLM call logging
    observability.log_llm_call(
        provider="OpenAI",
        model="gpt-3.5-turbo",
        prompt="Hello",
        response="Hi there!",
        tokens_used=10,
        latency=0.5
    )
    
    # Test retrieval logging
    observability.log_retrieval(
        query="test query",
        results_count=5,
        rag_type="Standard RAG",
        retrieval_time=0.2
    )
    
    # Test guardrails logging
    observability.log_guardrails_check(
        content="This is a test",
        is_toxic=False,
        is_nsfw=False,
        toxicity_score=0.1
    )
    
    # Display traces
    print("\n📊 Recent Traces:")
    for trace in observability.get_traces():
        print(f"  - {trace['name']}: {trace['status']} ({trace['duration']:.2f}s)")
    
    print("\n✅ Observability test complete!")

if __name__ == "__main__":
    test_observability()