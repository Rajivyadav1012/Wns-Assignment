"""
RAG Selector - Unified interface for all RAG variants
Step 10: Choose between Standard, Knowledge Graph, and Hybrid RAG
"""
from typing import Optional, Dict, List
from enum import Enum

from config.settings import settings
from utils.logger import setup_logger
from utils.database import vector_db
from modules.rag_standard import StandardRAG
from modules.rag_knowledge_graph import KnowledgeGraphRAG
from modules.rag_hybrid import HybridRAG

logger = setup_logger("rag_selector")

class RAGType(Enum):
    """RAG variant types"""
    STANDARD = "Standard RAG"
    KNOWLEDGE_GRAPH = "Knowledge Graph RAG"
    HYBRID = "Hybrid RAG"

class RAGSelector:
    """
    Unified interface to select and use different RAG variants
    """
    
    def __init__(
        self,
        rag_type: str = "Standard RAG",
        provider_name: str = "groq",
        model_name: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.3,
        **kwargs
    ):
        """
        Initialize RAG Selector
        
        Args:
            rag_type: Type of RAG ("Standard RAG", "Knowledge Graph RAG", "Hybrid RAG")
            provider_name: LLM provider
            model_name: Specific model
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score
            **kwargs: Additional parameters for specific RAG types
        """
        self.rag_type = rag_type
        self.provider_name = provider_name
        self.model_name = model_name
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        # Initialize the selected RAG
        self.rag_instance = None
        self._initialize_rag(**kwargs)
        
        logger.info(f"RAG Selector initialized with: {self.rag_type}")
    
    def _initialize_rag(self, **kwargs):
        """Initialize the selected RAG variant"""
        try:
            if self.rag_type == RAGType.STANDARD.value:
                self.rag_instance = StandardRAG(
                    provider_name=self.provider_name,
                    model_name=self.model_name,
                    top_k=self.top_k,
                    similarity_threshold=self.similarity_threshold
                )
                
            elif self.rag_type == RAGType.KNOWLEDGE_GRAPH.value:
                self.rag_instance = KnowledgeGraphRAG(
                    provider_name=self.provider_name,
                    model_name=self.model_name,
                    top_k=self.top_k,
                    similarity_threshold=self.similarity_threshold
                )
                
                # Build knowledge graph if documents exist
                if kwargs.get('build_kg', True):
                    self._build_knowledge_graph()
                    
            elif self.rag_type == RAGType.HYBRID.value:
                semantic_weight = kwargs.get('semantic_weight', 0.6)
                self.rag_instance = HybridRAG(
                    provider_name=self.provider_name,
                    model_name=self.model_name,
                    top_k=self.top_k,
                    similarity_threshold=self.similarity_threshold,
                    semantic_weight=semantic_weight
                )
                
                # Build BM25 index if documents exist
                if kwargs.get('build_bm25', True):
                    self._build_bm25_index()
            else:
                raise ValueError(f"Unknown RAG type: {self.rag_type}")
                
        except Exception as e:
            logger.error(f"Failed to initialize RAG: {e}")
            raise
    
    def _build_knowledge_graph(self):
        """Build knowledge graph for KG RAG"""
        try:
            logger.info("Building knowledge graph...")
            # Get all documents from vector DB
            query_embedding = self.rag_instance.embedding_generator.generate_single_embedding("init")
            all_docs = vector_db.search(query_embedding, top_k=100)
            
            if all_docs:
                self.rag_instance.build_graph_from_documents(all_docs)
            else:
                logger.warning("No documents found for knowledge graph")
        except Exception as e:
            logger.error(f"Failed to build knowledge graph: {e}")
    
    def _build_bm25_index(self):
        """Build BM25 index for Hybrid RAG"""
        try:
            logger.info("Building BM25 index...")
            # Get all documents from vector DB
            query_embedding = self.rag_instance.embedding_generator.generate_single_embedding("init")
            all_docs = vector_db.search(query_embedding, top_k=100)
            
            if all_docs:
                self.rag_instance.build_bm25_index(all_docs)
            else:
                logger.warning("No documents found for BM25 index")
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
    
    def switch_rag_type(
        self,
        rag_type: str,
        rebuild_indices: bool = True,
        **kwargs
    ):
        """
        Switch to a different RAG type
        
        Args:
            rag_type: New RAG type
            rebuild_indices: Whether to rebuild KG/BM25 indices
            **kwargs: Additional parameters
        """
        logger.info(f"Switching from {self.rag_type} to {rag_type}")
        self.rag_type = rag_type
        
        if rebuild_indices:
            self._initialize_rag(**kwargs)
        else:
            self._initialize_rag(build_kg=False, build_bm25=False, **kwargs)
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        return_sources: bool = True
    ) -> Dict:
        """
        Query using the selected RAG variant
        
        Args:
            query: User query
            top_k: Number of documents (overrides default)
            temperature: LLM temperature
            max_tokens: Maximum tokens
            return_sources: Return source documents
            
        Returns:
            Response dictionary with answer and metadata
        """
        if not self.rag_instance:
            return {
                "answer": "RAG not initialized",
                "sources": [],
                "error": "RAG instance is None"
            }
        
        try:
            response = self.rag_instance.query(
                query=query,
                top_k=top_k,
                temperature=temperature,
                max_tokens=max_tokens,
                return_sources=return_sources
            )
            
            # Add RAG type to response
            response['rag_type'] = self.rag_type
            
            return response
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "error": str(e),
                "rag_type": self.rag_type
            }
    
    def get_available_rag_types(self) -> List[str]:
        """Get list of available RAG types"""
        return [rag.value for rag in RAGType]
    
    def get_current_rag_type(self) -> str:
        """Get current RAG type"""
        return self.rag_type
    
    def get_stats(self) -> Dict:
        """Get statistics about the current RAG"""
        stats = {
            "rag_type": self.rag_type,
            "provider": self.provider_name,
            "model": self.model_name or "default",
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold
        }
        
        # Add RAG-specific stats
        if self.rag_type == RAGType.KNOWLEDGE_GRAPH.value and self.rag_instance:
            if hasattr(self.rag_instance, 'graph'):
                stats['kg_entities'] = self.rag_instance.graph.number_of_nodes()
                stats['kg_relationships'] = self.rag_instance.graph.number_of_edges()
        
        elif self.rag_type == RAGType.HYBRID.value and self.rag_instance:
            if hasattr(self.rag_instance, 'semantic_weight'):
                stats['semantic_weight'] = self.rag_instance.semantic_weight
                stats['bm25_weight'] = self.rag_instance.bm25_weight
        
        return stats

# Test function
def test_rag_selector():
    """Test RAG Selector with all variants"""
    print("\n" + "="*70)
    print("RAG SELECTOR - Testing All Variants")
    print("="*70 + "\n")
    
    # Check documents
    print("Checking for documents...")
    stats = vector_db.get_stats()
    if stats['total_vectors'] == 0:
        print("No documents found! Upload documents first.")
        return
    print(f"Found {stats['total_vectors']} vectors in database\n")
    
    test_query = "What is this document about?"
    
    # Test each RAG type
    for rag_type in ["Standard RAG", "Hybrid RAG", "Knowledge Graph RAG"]:
        print("\n" + "="*70)
        print(f"Testing: {rag_type}")
        print("="*70 + "\n")
        
        try:
            # Initialize selector
            print(f"Initializing {rag_type}...")
            selector = RAGSelector(
                rag_type=rag_type,
                provider_name="groq",
                model_name="llama-3.1-8b-instant",
                top_k=3,
                similarity_threshold=0.3,
                semantic_weight=0.6  # For Hybrid RAG
            )
            
            # Get stats
            rag_stats = selector.get_stats()
            print(f"Stats: {rag_stats}\n")
            
            # Query
            print(f"Query: {test_query}")
            response = selector.query(
                query=test_query,
                max_tokens=150,
                return_sources=True
            )
            
            print(f"\nRetrieved: {response.get('retrieved_count', 0)} documents")
            print(f"Time: {response.get('total_time', 0):.2f}s")
            print(f"Answer: {response.get('answer', 'N/A')[:200]}...")
            
            if response.get('sources'):
                print(f"\nSources ({len(response['sources'])}):")
                for i, src in enumerate(response['sources'][:2], 1):
                    method = src.get('method', 'unknown')
                    print(f"  {i}. {src['filename']} (method: {method})")
            
        except Exception as e:
            print(f"Failed to test {rag_type}: {e}")
    
    print("\n" + "="*70)
    print("RAG Selector tests complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_rag_selector()