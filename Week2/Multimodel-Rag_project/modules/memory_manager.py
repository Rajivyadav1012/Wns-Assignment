"""
Memory Manager
Step 12: Conversation history and context management
"""
import time
from typing import List, Dict, Optional
from datetime import datetime

from config.settings import settings
from utils.logger import setup_logger
from utils.database import memory_db
from utils.helpers import generate_session_id, count_tokens

logger = setup_logger("memory_manager")

class ConversationMemory:
    """Manage conversation history and context"""
    
    def __init__(
        self,
        session_id: Optional[str] = None,
        max_messages: int = 10,
        enable_memory: bool = True
    ):
        """
        Initialize conversation memory
        
        Args:
            session_id: Unique session identifier
            max_messages: Maximum messages to keep in context
            enable_memory: Whether to enable memory
        """
        self.session_id = session_id or generate_session_id()
        self.max_messages = max_messages
        self.enable_memory = enable_memory
        
        # In-memory cache for fast access
        self.message_cache = []
        
        logger.info(f"Memory initialized for session {self.session_id[:8]} (max: {max_messages} messages)")
    
    def add_message(
        self,
        role: str,
        content: str,
        model_used: Optional[str] = None,
        tokens_used: Optional[int] = None
    ):
        """
        Add a message to conversation history
        
        Args:
            role: Message role ('user' or 'assistant')
            content: Message content
            model_used: Model that generated the message
            tokens_used: Number of tokens used
        """
        if not self.enable_memory:
            return
        
        try:
            # Add to database
            memory_db.add_message(
                session_id=self.session_id,
                role=role,
                content=content,
                model_used=model_used,
                tokens_used=tokens_used
            )
            
            # Add to cache
            self.message_cache.append({
                'role': role,
                'content': content,
                'timestamp': datetime.now(),
                'model_used': model_used,
                'tokens_used': tokens_used
            })
            
            # Trim cache if needed
            if len(self.message_cache) > self.max_messages:
                self.message_cache = self.message_cache[-self.max_messages:]
            
            logger.debug(f"Added {role} message to memory")
            
        except Exception as e:
            logger.error(f"Failed to add message: {e}")
    
    def get_recent_messages(
        self,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Get recent messages from history
        
        Args:
            limit: Maximum number of messages (uses max_messages if None)
            
        Returns:
            List of recent messages
        """
        if not self.enable_memory:
            return []
        
        limit = limit or self.max_messages
        
        try:
            # Try cache first
            if self.message_cache:
                return self.message_cache[-limit:]
            
            # Fall back to database
            messages = memory_db.get_recent_messages(self.session_id, limit)
            
            # Convert to dict format
            return [
                {
                    'role': msg.role,
                    'content': msg.content,
                    'timestamp': msg.timestamp,
                    'model_used': msg.model_used,
                    'tokens_used': msg.tokens_used
                }
                for msg in messages
            ]
            
        except Exception as e:
            logger.error(f"Failed to get messages: {e}")
            return []
    
    def format_for_llm(
        self,
        include_system: bool = True,
        system_prompt: Optional[str] = None
    ) -> List[Dict]:
        """
        Format conversation history for LLM
        
        Args:
            include_system: Whether to include system prompt
            system_prompt: Custom system prompt
            
        Returns:
            List of messages in LLM format
        """
        messages = []
        
        if include_system:
            messages.append({
                'role': 'system',
                'content': system_prompt or settings.SYSTEM_PROMPT
            })
        
        recent = self.get_recent_messages()
        for msg in recent:
            messages.append({
                'role': msg['role'],
                'content': msg['content']
            })
        
        return messages
    
    def get_context_summary(self) -> str:
        """
        Get a summary of the conversation context
        
        Returns:
            Formatted summary string
        """
        messages = self.get_recent_messages()
        
        if not messages:
            return "No conversation history"
        
        summary_parts = []
        for msg in messages:
            role = msg['role'].capitalize()
            content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
            summary_parts.append(f"{role}: {content}")
        
        return "\n".join(summary_parts)
    
    def clear_history(self):
        """Clear conversation history"""
        try:
            memory_db.clear_session(self.session_id)
            self.message_cache = []
            logger.info(f"Cleared history for session {self.session_id[:8]}")
        except Exception as e:
            logger.error(f"Failed to clear history: {e}")
    
    def get_statistics(self) -> Dict:
        """
        Get conversation statistics
        
        Returns:
            Dictionary with stats
        """
        messages = self.get_recent_messages(limit=1000)  # Get all
        
        user_messages = [m for m in messages if m['role'] == 'user']
        assistant_messages = [m for m in messages if m['role'] == 'assistant']
        
        total_tokens = sum(m.get('tokens_used', 0) for m in messages if m.get('tokens_used'))
        
        return {
            'session_id': self.session_id,
            'total_messages': len(messages),
            'user_messages': len(user_messages),
            'assistant_messages': len(assistant_messages),
            'total_tokens': total_tokens,
            'in_context_messages': len(self.get_recent_messages())
        }

class RAGWithMemory:
    """RAG system with conversation memory"""
    
    def __init__(
        self,
        rag_instance,
        session_id: Optional[str] = None,
        max_messages: int = 10,
        enable_memory: bool = True
    ):
        """
        Initialize RAG with memory
        
        Args:
            rag_instance: Existing RAG instance
            session_id: Session identifier
            max_messages: Max messages to keep
            enable_memory: Enable memory
        """
        self.rag = rag_instance
        self.memory = ConversationMemory(
            session_id=session_id,
            max_messages=max_messages,
            enable_memory=enable_memory
        )
        
        logger.info("RAG with memory initialized")
    
    def query(
        self,
        query: str,
        use_memory: bool = True,
        **kwargs
    ) -> Dict:
        """
        Query with conversation memory
        
        Args:
            query: User query
            use_memory: Whether to use conversation history
            **kwargs: Additional RAG arguments
            
        Returns:
            Response dictionary
        """
        start_time = time.time()
        
        try:
            # Add user message to memory
            if use_memory:
                self.memory.add_message('user', query)
            
            # Get RAG response
            response = self.rag.query(query, **kwargs)
            
            # Add assistant message to memory
            if use_memory and response.get('answer'):
                self.memory.add_message(
                    'assistant',
                    response['answer'],
                    model_used=getattr(self.rag, 'model_name', None)
                )
            
            # Add memory stats to response
            response['memory_stats'] = self.memory.get_statistics()
            response['session_id'] = self.memory.session_id
            
            return response
            
        except Exception as e:
            logger.error(f"Query with memory failed: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "error": str(e)
            }
    
    def chat(
        self,
        query: str,
        include_history_in_context: bool = False,
        **kwargs
    ) -> Dict:
        """
        Chat with conversation history context
        
        Args:
            query: User query
            include_history_in_context: Add history to retrieval context
            **kwargs: Additional arguments
            
        Returns:
            Response dictionary
        """
        # If we want to include history in context, prepend it to the query
        if include_history_in_context:
            context_summary = self.memory.get_context_summary()
            enhanced_query = f"Previous conversation:\n{context_summary}\n\nCurrent question: {query}"
        else:
            enhanced_query = query
        
        return self.query(enhanced_query, use_memory=True, **kwargs)
    
    def get_conversation_history(self) -> List[Dict]:
        """Get full conversation history"""
        return self.memory.get_recent_messages(limit=1000)
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear_history()

# Test function
def test_memory_manager():
    """Test memory manager"""
    print("\n" + "="*70)
    print("TESTING MEMORY MANAGER")
    print("="*70 + "\n")
    
    # Test basic memory
    print("1. Testing conversation memory...")
    memory = ConversationMemory(max_messages=5)
    print(f"   Session ID: {memory.session_id[:16]}...\n")
    
    # Add some messages
    conversations = [
        ("user", "What is RAG?"),
        ("assistant", "RAG stands for Retrieval-Augmented Generation."),
        ("user", "How does it work?"),
        ("assistant", "It retrieves relevant documents and uses them to generate answers."),
        ("user", "What are the benefits?"),
    ]
    
    for role, content in conversations:
        memory.add_message(role, content)
        print(f"   Added: {role} - {content[:50]}...")
    
    print(f"\n   Total messages in memory: {len(memory.get_recent_messages())}")
    
    # Test statistics
    print("\n2. Testing statistics...")
    stats = memory.get_statistics()
    print(f"   User messages: {stats['user_messages']}")
    print(f"   Assistant messages: {stats['assistant_messages']}")
    print(f"   Total messages: {stats['total_messages']}")
    
    # Test context summary
    print("\n3. Testing context summary...")
    summary = memory.get_context_summary()
    print(f"   Summary:\n{summary}\n")
    
    # Test with RAG
    print("\n4. Testing RAG with memory...")
    try:
        from modules.rag_standard import StandardRAG
        from utils.database import vector_db
        
        stats = vector_db.get_stats()
        if stats['total_vectors'] == 0:
            print("   No documents - skipping RAG test")
        else:
            rag = StandardRAG("groq", top_k=2, similarity_threshold=0.3)
            rag_with_memory = RAGWithMemory(rag, max_messages=5)
            
            # Simulate a conversation
            queries = [
                "What is this document about?",
                "Can you provide more details?"
            ]
            
            for query in queries:
                print(f"\n   Query: {query}")
                response = rag_with_memory.query(query, max_tokens=100)
                print(f"   Answer: {response['answer'][:100]}...")
                print(f"   Messages in context: {response['memory_stats']['in_context_messages']}")
            
            # Show conversation history
            print("\n   Conversation history:")
            history = rag_with_memory.get_conversation_history()
            for i, msg in enumerate(history[-4:], 1):
                print(f"      {i}. {msg['role']}: {msg['content'][:80]}...")
    
    except Exception as e:
        print(f"   RAG test failed: {e}")
    
    print("\n" + "="*70)
    print("Memory manager tests complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_memory_manager()