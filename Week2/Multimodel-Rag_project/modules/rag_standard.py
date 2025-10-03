"""
Standard RAG Implementation
Step 7: Retrieval-Augmented Generation
"""
import time
from typing import List, Dict, Optional, Tuple
import numpy as np

from config.settings import settings
from utils.logger import setup_logger
from utils.database import vector_db
from utils.observability import observability
from modules.llm_providers import LLMFactory
from modules.document_processor import EmbeddingGenerator

logger = setup_logger("rag_standard")

class StandardRAG:
    """Standard RAG implementation with semantic search"""
    
    def __init__(
        self,
        provider_name: str = "groq",
        model_name: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize Standard RAG
        
        Args:
            provider_name: LLM provider (groq, openai, etc.)
            model_name: Specific model to use
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score
        """
        self.provider_name = provider_name
        self.model_name = model_name
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        # Initialize components
        self.llm = LLMFactory.create_provider(provider_name, model_name)
        self.embedding_generator = EmbeddingGenerator()
        
        logger.info(f"✅ Standard RAG initialized with {provider_name}")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: User query
            top_k: Number of results (overrides default)
            filters: Optional filters (e.g., by document_id)
            
        Returns:
            List of relevant document chunks
        """
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_single_embedding(query)
            
            # Search vector database
            k = top_k or self.top_k
            results = vector_db.search(query_embedding, top_k=k)
            
            # Filter by similarity threshold
            filtered_results = [
                r for r in results 
                if r.get('similarity', 0) >= self.similarity_threshold
            ]
            
            # Apply additional filters if provided
            if filters:
                if 'document_id' in filters:
                    filtered_results = [
                        r for r in filtered_results
                        if r.get('document_id') == filters['document_id']
                    ]
            
            retrieval_time = time.time() - start_time
            
            # Log retrieval
            observability.log_retrieval(
                query=query,
                results_count=len(filtered_results),
                rag_type="Standard RAG",
                retrieval_time=retrieval_time
            )
            
            logger.info(f"🔍 Retrieved {len(filtered_results)} documents in {retrieval_time:.2f}s")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}")
            return []
    
    def format_context(self, retrieved_docs: List[Dict]) -> str:
        """
        Format retrieved documents into context string
        
        Args:
            retrieved_docs: List of retrieved document chunks
            
        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            filename = doc.get('filename', 'Unknown')
            chunk_text = doc.get('chunk_text', doc.get('text', ''))
            similarity = doc.get('similarity', 0)
            
            context_parts.append(
                f"[Document {i}] {filename} (relevance: {similarity:.2f})\n{chunk_text}"
            )
        
        return "\n\n".join(context_parts)
    
    def generate_prompt(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Generate prompt for LLM
        
        Args:
            query: User query
            context: Retrieved context
            system_prompt: Optional system prompt
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system = system_prompt or settings.SYSTEM_PROMPT
        
        user_prompt = settings.RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=query
        )
        
        return system, user_prompt
    
    def generate_answer(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        Generate answer using LLM
        
        Args:
            query: User query
            context: Retrieved context
            system_prompt: Optional system prompt
            temperature: LLM temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated answer
        """
        try:
            system, user_prompt = self.generate_prompt(query, context, system_prompt)
            
            answer = self.llm.generate(
                prompt=user_prompt,
                system_prompt=system,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return answer
            
        except Exception as e:
            logger.error(f"❌ Answer generation failed: {e}")
            return f"Error generating answer: {str(e)}"
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        return_sources: bool = True
    ) -> Dict:
        """
        Complete RAG pipeline: retrieve + generate
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            temperature: LLM temperature
            max_tokens: Maximum tokens
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary with answer and optional sources
        """
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Processing query: {query}")
            
            # Step 1: Retrieve relevant documents
            retrieved_docs = self.retrieve(query, top_k=top_k)
            
            if not retrieved_docs:
                return {
                    "answer": "I don't have enough information to answer this question based on the available documents.",
                    "sources": [],
                    "retrieved_count": 0,
                    "total_time": time.time() - start_time
                }
            
            # Step 2: Format context
            context = self.format_context(retrieved_docs)
            
            # Step 3: Generate answer
            answer = self.generate_answer(
                query=query,
                context=context,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            total_time = time.time() - start_time
            
            # Prepare response
            response = {
                "answer": answer,
                "retrieved_count": len(retrieved_docs),
                "total_time": total_time
            }
            
            if return_sources:
                sources = [
                    {
                        "filename": doc.get('filename', 'Unknown'),
                        "chunk_id": doc.get('chunk_id', 0),
                        "similarity": doc.get('similarity', 0),
                        "text": doc.get('chunk_text', '')[:200] + "..."
                    }
                    for doc in retrieved_docs
                ]
                response["sources"] = sources
            
            logger.info(f"✅ Query completed in {total_time:.2f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "retrieved_count": 0,
                "total_time": time.time() - start_time
            }
    
    def chat(
        self,
        query: str,
        conversation_history: Optional[List[Dict]] = None,
        top_k: Optional[int] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict:
        """
        RAG with conversation history support
        
        Args:
            query: User query
            conversation_history: List of previous messages
            top_k: Number of documents to retrieve
            temperature: LLM temperature
            max_tokens: Maximum tokens
            
        Returns:
            Response dictionary
        """
        # For now, use the same query method
        # In future iterations, we can incorporate conversation history
        # into the retrieval and generation process
        
        return self.query(
            query=query,
            top_k=top_k,
            temperature=temperature,
            max_tokens=max_tokens,
            return_sources=True
        )

# ==================== Test Function ====================

def test_standard_rag():
    """Test Standard RAG"""
    print("\n" + "="*70)
    print("🤖 TESTING STANDARD RAG")
    print("="*70 + "\n")
    
    # Check if documents exist
    print("1️⃣  Checking for documents...")
    stats = vector_db.get_stats()
    if stats['total_vectors'] == 0:
        print("   ⚠️  No documents found! Please upload documents first.")
        print("   Run: python -m modules.document_processor")
        return
    print(f"   ✅ Found {stats['total_vectors']} vectors in database\n")
    
    # Initialize RAG
    print("2️⃣  Initializing Standard RAG...")
    try:
        rag = StandardRAG(
            provider_name="groq",
            model_name="llama-3.1-8b-instant",
            top_k=3,
            similarity_threshold=0.3
        )
        print("   ✅ RAG initialized\n")
    except Exception as e:
        print(f"   ❌ Failed to initialize RAG: {e}")
        return
    
    # Test retrieval only
    print("3️⃣  Testing retrieval...")
    test_query = "What is this document about?"
    retrieved = rag.retrieve(test_query)
    print(f"   ✅ Retrieved {len(retrieved)} documents")
    for i, doc in enumerate(retrieved[:2], 1):
        print(f"      {i}. {doc.get('filename', 'Unknown')} (similarity: {doc.get('similarity', 0):.3f})")
    print()
    
    # Test full RAG pipeline
    print("4️⃣  Testing full RAG pipeline...")
    queries = [
        "What is this document about?",
        "Summarize the main points"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n   Query {i}: {query}")
        print("   " + "-"*60)
        
        response = rag.query(query, top_k=3, max_tokens=200)
        
        print(f"   📊 Retrieved: {response['retrieved_count']} documents")
        print(f"   ⏱️  Time: {response['total_time']:.2f}s")
        print(f"   💬 Answer:\n      {response['answer'][:200]}...")
        
        if response.get('sources'):
            print(f"\n   📚 Sources:")
            for j, source in enumerate(response['sources'][:2], 1):
                print(f"      {j}. {source['filename']} (relevance: {source['similarity']:.2f})")
    
    print("\n" + "="*70)
    print("✅ Standard RAG tests complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_standard_rag()