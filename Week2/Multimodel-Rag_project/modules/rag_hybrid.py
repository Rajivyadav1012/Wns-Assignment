"""
Hybrid RAG Implementation
Step 9: Combines semantic search (vector) + keyword search (BM25)
"""
import time
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
import numpy as np

from config.settings import settings
from utils.logger import setup_logger
from utils.database import vector_db
from utils.observability import observability
from modules.llm_providers import LLMFactory
from modules.document_processor import EmbeddingGenerator

logger = setup_logger("rag_hybrid")

class HybridRAG:
    """Hybrid RAG with semantic + keyword search"""
    
    def __init__(
        self,
        provider_name: str = "groq",
        model_name: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        semantic_weight: float = 0.5
    ):
        """
        Initialize Hybrid RAG
        
        Args:
            provider_name: LLM provider
            model_name: Specific model
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score
            semantic_weight: Weight for semantic search (0-1), BM25 gets (1-weight)
        """
        self.provider_name = provider_name
        self.model_name = model_name
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.semantic_weight = semantic_weight
        self.bm25_weight = 1 - semantic_weight
        
        # Initialize components
        self.llm = LLMFactory.create_provider(provider_name, model_name)
        self.embedding_generator = EmbeddingGenerator()
        
        # BM25 index (will be built from documents)
        self.bm25 = None
        self.bm25_docs = []
        self.bm25_metadata = []
        
        logger.info(f"✅ Hybrid RAG initialized (semantic: {semantic_weight:.1f}, keyword: {self.bm25_weight:.1f})")
    
    def build_bm25_index(self, documents: List[Dict]):
        """
        Build BM25 index from documents
        
        Args:
            documents: List of document chunks with metadata
        """
        logger.info("🔨 Building BM25 keyword index...")
        start_time = time.time()
        
        try:
            self.bm25_docs = []
            self.bm25_metadata = []
            
            for doc in documents:
                text = doc.get('chunk_text', doc.get('text', ''))
                if text:
                    # Tokenize (simple split by whitespace and lowercase)
                    tokens = text.lower().split()
                    self.bm25_docs.append(tokens)
                    self.bm25_metadata.append(doc)
            
            if self.bm25_docs:
                self.bm25 = BM25Okapi(self.bm25_docs)
                elapsed = time.time() - start_time
                logger.info(f"✅ BM25 index built with {len(self.bm25_docs)} documents in {elapsed:.2f}s")
            else:
                logger.warning("⚠️ No documents to index")
                
        except Exception as e:
            logger.error(f"❌ Failed to build BM25 index: {e}")
    
    def bm25_search(self, query: str, top_k: int) -> List[Dict]:
        """
        Perform BM25 keyword search
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of documents with BM25 scores
        """
        if not self.bm25 or not self.bm25_docs:
            logger.warning("⚠️ BM25 index not built")
            return []
        
        try:
            # Tokenize query
            query_tokens = query.lower().split()
            
            # Get BM25 scores
            scores = self.bm25.get_scores(query_tokens)
            
            # Get top-k indices
            top_indices = np.argsort(scores)[-top_k:][::-1]
            
            # Prepare results
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include if score > 0
                    doc = self.bm25_metadata[idx].copy()
                    doc['bm25_score'] = float(scores[idx])
                    doc['retrieval_method'] = 'bm25'
                    results.append(doc)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ BM25 search failed: {e}")
            return []
    
    def semantic_search(self, query: str, top_k: int) -> List[Dict]:
        """
        Perform semantic vector search
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of documents with similarity scores
        """
        try:
            query_embedding = self.embedding_generator.generate_single_embedding(query)
            results = vector_db.search(query_embedding, top_k=top_k)
            
            for doc in results:
                doc['retrieval_method'] = 'semantic'
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Semantic search failed: {e}")
            return []
    
    def reciprocal_rank_fusion(
        self,
        semantic_results: List[Dict],
        bm25_results: List[Dict],
        k: int = 60
    ) -> List[Dict]:
        """
        Combine results using Reciprocal Rank Fusion (RRF)
        
        Args:
            semantic_results: Results from semantic search
            bm25_results: Results from BM25 search
            k: RRF parameter (typically 60)
            
        Returns:
            Fused and ranked results
        """
        # Create doc_id for each result
        def get_doc_key(doc):
            return f"{doc.get('document_id', '')}_{doc.get('chunk_id', '')}"
        
        # Calculate RRF scores
        rrf_scores = {}
        
        # Add semantic results
        for rank, doc in enumerate(semantic_results, start=1):
            doc_key = get_doc_key(doc)
            score = self.semantic_weight / (k + rank)
            rrf_scores[doc_key] = {
                'score': score,
                'doc': doc,
                'semantic_rank': rank,
                'bm25_rank': None
            }
        
        # Add BM25 results
        for rank, doc in enumerate(bm25_results, start=1):
            doc_key = get_doc_key(doc)
            score = self.bm25_weight / (k + rank)
            
            if doc_key in rrf_scores:
                rrf_scores[doc_key]['score'] += score
                rrf_scores[doc_key]['bm25_rank'] = rank
                rrf_scores[doc_key]['doc']['retrieval_method'] = 'hybrid'
            else:
                rrf_scores[doc_key] = {
                    'score': score,
                    'doc': doc,
                    'semantic_rank': None,
                    'bm25_rank': rank
                }
        
        # Sort by RRF score
        sorted_results = sorted(
            rrf_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        # Extract documents with RRF scores
        final_results = []
        for item in sorted_results:
            doc = item['doc'].copy()
            doc['rrf_score'] = item['score']
            doc['semantic_rank'] = item['semantic_rank']
            doc['bm25_rank'] = item['bm25_rank']
            final_results.append(doc)
        
        return final_results
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Hybrid retrieval: semantic + BM25 with RRF
        
        Args:
            query: User query
            top_k: Number of results
            
        Returns:
            Fused results
        """
        start_time = time.time()
        k = top_k or self.top_k
        
        try:
            # Perform both searches
            semantic_results = self.semantic_search(query, top_k=k * 2)
            bm25_results = self.bm25_search(query, top_k=k * 2)
            
            logger.info(f"🔍 Semantic: {len(semantic_results)}, BM25: {len(bm25_results)} results")
            
            # Fuse results
            fused_results = self.reciprocal_rank_fusion(
                semantic_results,
                bm25_results
            )
            
            # Filter by similarity threshold and limit to top_k
            filtered_results = [
                r for r in fused_results
                if r.get('similarity', 1) >= self.similarity_threshold
            ][:k]
            
            retrieval_time = time.time() - start_time
            
            observability.log_retrieval(
                query=query,
                results_count=len(filtered_results),
                rag_type="Hybrid RAG",
                retrieval_time=retrieval_time
            )
            
            logger.info(f"✅ Hybrid retrieval: {len(filtered_results)} documents in {retrieval_time:.2f}s")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"❌ Hybrid retrieval failed: {e}")
            return []
    
    def format_context(self, retrieved_docs: List[Dict]) -> str:
        """Format retrieved documents into context"""
        if not retrieved_docs:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            filename = doc.get('filename', 'Unknown')
            chunk_text = doc.get('chunk_text', doc.get('text', ''))
            method = doc.get('retrieval_method', 'unknown')
            rrf_score = doc.get('rrf_score', 0)
            
            context_parts.append(
                f"[Document {i}] {filename} (RRF: {rrf_score:.3f}, method: {method})\n{chunk_text}"
            )
        
        return "\n\n".join(context_parts)
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        return_sources: bool = True
    ) -> Dict:
        """
        Complete Hybrid RAG pipeline
        
        Args:
            query: User query
            top_k: Number of documents
            temperature: LLM temperature
            max_tokens: Maximum tokens
            return_sources: Return sources
            
        Returns:
            Response dictionary
        """
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Processing hybrid query: {query}")
            
            # Retrieve with hybrid search
            retrieved_docs = self.retrieve(query, top_k=top_k)
            
            if not retrieved_docs:
                return {
                    "answer": "I don't have enough information to answer this question.",
                    "sources": [],
                    "retrieved_count": 0,
                    "total_time": time.time() - start_time
                }
            
            # Format context
            context = self.format_context(retrieved_docs)
            
            # Generate answer
            system_prompt = settings.SYSTEM_PROMPT
            user_prompt = settings.RAG_PROMPT_TEMPLATE.format(
                context=context,
                question=query
            )
            
            answer = self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            total_time = time.time() - start_time
            
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
                        "method": doc.get('retrieval_method', 'unknown'),
                        "rrf_score": doc.get('rrf_score', 0),
                        "text": doc.get('chunk_text', '')[:200] + "..."
                    }
                    for doc in retrieved_docs
                ]
                response["sources"] = sources
            
            logger.info(f"✅ Hybrid query completed in {total_time:.2f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Hybrid query failed: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "retrieved_count": 0,
                "total_time": time.time() - start_time
            }

# Test function
def test_hybrid_rag():
    """Test Hybrid RAG"""
    print("\n" + "="*70)
    print("⚡ TESTING HYBRID RAG")
    print("="*70 + "\n")
    
    # Check documents
    print("1️⃣  Checking for documents...")
    stats = vector_db.get_stats()
    if stats['total_vectors'] == 0:
        print("   No documents found!")
        return
    print(f"   Found {stats['total_vectors']} vectors\n")
    
    # Initialize Hybrid RAG
    print("2️⃣  Initializing Hybrid RAG...")
    try:
        hybrid_rag = HybridRAG(
            provider_name="groq",
            model_name="llama-3.1-8b-instant",
            top_k=3,
            similarity_threshold=0.3,  # Lower threshold for test documents
            semantic_weight=0.6  # 60% semantic, 40% keyword
        )
        print("   Hybrid RAG initialized\n")
    except Exception as e:
        print(f"   Failed: {e}")
        return
    
    # Build BM25 index
    print("3️⃣  Building BM25 index...")
    query_embedding = hybrid_rag.embedding_generator.generate_single_embedding("test")
    all_docs = vector_db.search(query_embedding, top_k=100)
    hybrid_rag.build_bm25_index(all_docs)
    print()
    
    # Test query
    print("4️⃣  Testing hybrid query...")
    query = "What is this document about?"
    print(f"   Query: {query}\n")
    
    response = hybrid_rag.query(query, top_k=3, max_tokens=200)
    
    print(f"   Retrieved: {response['retrieved_count']} documents")
    print(f"   Time: {response['total_time']:.2f}s")
    print(f"   Answer: {response['answer'][:200]}...")
    
    if response.get('sources'):
        print(f"\n   Sources:")
        for i, src in enumerate(response['sources'], 1):
            print(f"      {i}. {src['filename']} (method: {src['method']}, RRF: {src['rrf_score']:.3f})")
    
    print("\n" + "="*70)
    print("Hybrid RAG tests complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_hybrid_rag()