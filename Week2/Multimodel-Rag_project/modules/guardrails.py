"""
Guardrails Module
Step 13: Content safety filters (toxicity, NSFW detection)
"""
import time
from typing import Dict, Optional, Tuple
from detoxify import Detoxify

from config.settings import settings
from utils.logger import setup_logger
from utils.observability import observability

logger = setup_logger("guardrails")

class ContentGuardrails:
    """Content safety guardrails using Detoxify"""
    
    def __init__(
        self,
        toxicity_threshold: float = 0.7,
        enable_guardrails: bool = True
    ):
        """
        Initialize content guardrails
        
        Args:
            toxicity_threshold: Threshold for toxicity detection (0-1)
            enable_guardrails: Whether guardrails are enabled
        """
        self.toxicity_threshold = toxicity_threshold
        self.enable_guardrails = enable_guardrails
        
        # Initialize Detoxify model
        self.model = None
        if enable_guardrails:
            try:
                logger.info("Loading toxicity detection model...")
                self.model = Detoxify('original')
                logger.info("Guardrails enabled with Detoxify model")
            except Exception as e:
                logger.error(f"Failed to load Detoxify model: {e}")
                self.enable_guardrails = False
        else:
            logger.warning("Guardrails disabled")
    
    def check_toxicity(self, text: str) -> Dict:
        """
        Check text for toxicity
        
        Args:
            text: Text to check
            
        Returns:
            Dictionary with toxicity scores
        """
        if not self.enable_guardrails or not self.model:
            return {
                'is_toxic': False,
                'scores': {},
                'checked': False
            }
        
        try:
            start_time = time.time()
            
            # Get toxicity predictions
            results = self.model.predict(text)
            
            # Check against threshold
            is_toxic = any(
                score > self.toxicity_threshold 
                for score in results.values()
            )
            
            elapsed = time.time() - start_time
            
            # Log to observability
            observability.log_guardrails_check(
                content=text,
                is_toxic=is_toxic,
                is_nsfw=results.get('sexual_explicit', 0) > self.toxicity_threshold,
                toxicity_score=results.get('toxicity', 0)
            )
            
            logger.debug(f"Toxicity check completed in {elapsed:.2f}s")
            
            return {
                'is_toxic': is_toxic,
                'scores': results,
                'checked': True,
                'check_time': elapsed
            }
            
        except Exception as e:
            logger.error(f"Toxicity check failed: {e}")
            return {
                'is_toxic': False,
                'scores': {},
                'checked': False,
                'error': str(e)
            }
    
    def check_input(self, text: str) -> Tuple[bool, str, Dict]:
        """
        Check if input should be allowed
        
        Args:
            text: Input text to check
            
        Returns:
            Tuple of (is_allowed, message, details)
        """
        if not self.enable_guardrails:
            return True, "", {}
        
        result = self.check_toxicity(text)
        
        if result['is_toxic']:
            scores = result.get('scores', {})
            violations = [
                category for category, score in scores.items()
                if score > self.toxicity_threshold
            ]
            
            message = f"Your message was flagged for inappropriate content: {', '.join(violations)}"
            logger.warning(f"Input blocked: {violations}")
            
            return False, message, result
        
        return True, "", result
    
    def check_output(self, text: str) -> Tuple[bool, str, Dict]:
        """
        Check if output should be shown
        
        Args:
            text: Output text to check
            
        Returns:
            Tuple of (is_allowed, message, details)
        """
        if not self.enable_guardrails:
            return True, "", {}
        
        result = self.check_toxicity(text)
        
        if result['is_toxic']:
            message = "The generated response contained inappropriate content and was blocked."
            logger.warning("Output blocked due to toxicity")
            
            return False, message, result
        
        return True, "", result
    
    def get_safe_categories(self, scores: Dict) -> Dict[str, str]:
        """
        Categorize toxicity scores
        
        Args:
            scores: Toxicity scores
            
        Returns:
            Dictionary with safety levels
        """
        categories = {}
        
        for category, score in scores.items():
            if score < 0.3:
                level = "safe"
            elif score < 0.6:
                level = "moderate"
            elif score < 0.8:
                level = "high"
            else:
                level = "very_high"
            
            categories[category] = {
                'score': float(score),
                'level': level
            }
        
        return categories

class RAGWithGuardrails:
    """RAG system with content safety guardrails"""
    
    def __init__(
        self,
        rag_instance,
        toxicity_threshold: float = 0.7,
        enable_guardrails: bool = True
    ):
        """
        Initialize RAG with guardrails
        
        Args:
            rag_instance: Existing RAG instance
            toxicity_threshold: Toxicity threshold
            enable_guardrails: Enable guardrails
        """
        self.rag = rag_instance
        self.guardrails = ContentGuardrails(
            toxicity_threshold=toxicity_threshold,
            enable_guardrails=enable_guardrails
        )
        
        logger.info("RAG with guardrails initialized")
    
    def query(
        self,
        query: str,
        check_input: bool = True,
        check_output: bool = True,
        **kwargs
    ) -> Dict:
        """
        Query with guardrails
        
        Args:
            query: User query
            check_input: Check input for toxicity
            check_output: Check output for toxicity
            **kwargs: Additional RAG arguments
            
        Returns:
            Response dictionary
        """
        start_time = time.time()
        
        # Check input
        if check_input:
            is_allowed, message, details = self.guardrails.check_input(query)
            
            if not is_allowed:
                return {
                    'answer': message,
                    'blocked': True,
                    'reason': 'input_violation',
                    'details': details,
                    'total_time': time.time() - start_time
                }
        
        # Process query
        try:
            response = self.rag.query(query, **kwargs)
            
            # Check output
            if check_output and response.get('answer'):
                is_allowed, message, details = self.guardrails.check_output(response['answer'])
                
                if not is_allowed:
                    response['answer'] = message
                    response['blocked'] = True
                    response['reason'] = 'output_violation'
                    response['details'] = details
            
            response['total_time'] = time.time() - start_time
            response['guardrails_enabled'] = self.guardrails.enable_guardrails
            
            return response
            
        except Exception as e:
            logger.error(f"Query with guardrails failed: {e}")
            return {
                'answer': f"Error: {str(e)}",
                'error': str(e),
                'total_time': time.time() - start_time
            }

# Test function
def test_guardrails():
    """Test guardrails functionality"""
    print("\n" + "="*70)
    print("TESTING GUARDRAILS")
    print("="*70 + "\n")
    
    # Initialize guardrails
    print("1. Initializing guardrails...")
    guardrails = ContentGuardrails(toxicity_threshold=0.7)
    
    if not guardrails.enable_guardrails:
        print("   Guardrails not enabled - Detoxify model failed to load")
        return
    
    print("   Guardrails enabled\n")
    
    # Test safe content
    print("2. Testing safe content...")
    safe_text = "Hello, how are you today? This is a friendly message."
    result = guardrails.check_toxicity(safe_text)
    print(f"   Text: {safe_text}")
    print(f"   Is toxic: {result['is_toxic']}")
    print(f"   Toxicity score: {result['scores'].get('toxicity', 0):.3f}\n")
    
    # Test potentially toxic content
    print("3. Testing potentially toxic content...")
    toxic_text = "You are stupid and I hate you!"
    result = guardrails.check_toxicity(toxic_text)
    print(f"   Text: {toxic_text}")
    print(f"   Is toxic: {result['is_toxic']}")
    print(f"   Toxicity score: {result['scores'].get('toxicity', 0):.3f}")
    
    if result['is_toxic']:
        print(f"   Violations detected:")
        for category, score in result['scores'].items():
            if score > 0.7:
                print(f"      - {category}: {score:.3f}")
    print()
    
    # Test with RAG
    print("4. Testing RAG with guardrails...")
    try:
        from modules.rag_standard import StandardRAG
        from utils.database import vector_db
        
        stats = vector_db.get_stats()
        if stats['total_vectors'] == 0:
            print("   No documents - skipping RAG test")
        else:
            rag = StandardRAG("groq", top_k=2, similarity_threshold=0.3)
            rag_with_guardrails = RAGWithGuardrails(rag, toxicity_threshold=0.7)
            
            # Test safe query
            print("\n   Safe query test:")
            query = "What is this document about?"
            print(f"   Query: {query}")
            response = rag_with_guardrails.query(query, max_tokens=100)
            print(f"   Blocked: {response.get('blocked', False)}")
            print(f"   Answer: {response['answer'][:100]}...")
            
            # Test toxic query
            print("\n   Toxic query test:")
            toxic_query = "I hate this stupid document!"
            print(f"   Query: {toxic_query}")
            response = rag_with_guardrails.query(toxic_query, max_tokens=100)
            print(f"   Blocked: {response.get('blocked', False)}")
            if response.get('blocked'):
                print(f"   Reason: {response.get('reason')}")
            print(f"   Answer: {response['answer'][:100]}...")
    
    except Exception as e:
        print(f"   RAG test failed: {e}")
    
    print("\n" + "="*70)
    print("Guardrails tests complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_guardrails()