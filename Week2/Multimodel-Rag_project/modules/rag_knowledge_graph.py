"""
Knowledge Graph RAG Implementation
Step 8: Entity-based retrieval using knowledge graphs
"""
import time
from typing import List, Dict, Optional, Set, Tuple
import networkx as nx
import spacy
from collections import defaultdict, Counter

from config.settings import settings
from utils.logger import setup_logger
from utils.database import vector_db
from utils.observability import observability
from modules.llm_providers import LLMFactory
from modules.document_processor import EmbeddingGenerator

logger = setup_logger("rag_kg")

class KnowledgeGraphRAG:
    """Knowledge Graph RAG with entity extraction and graph-based retrieval"""
    
    def __init__(
        self,
        provider_name: str = "groq",
        model_name: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize Knowledge Graph RAG
        
        Args:
            provider_name: LLM provider
            model_name: Specific model
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
        
        # Load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ Loaded spaCy model for entity extraction")
        except Exception as e:
            logger.error(f"❌ Failed to load spaCy: {e}")
            raise
        
        # Initialize knowledge graph
        self.graph = nx.DiGraph()
        self.entity_to_chunks = defaultdict(set)  # Maps entities to chunk IDs
        
        logger.info(f"✅ Knowledge Graph RAG initialized with {provider_name}")
    
    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract named entities from text
        
        Args:
            text: Input text
            
        Returns:
            List of (entity_text, entity_type) tuples
        """
        try:
            doc = self.nlp(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            return entities
        except Exception as e:
            logger.error(f"❌ Entity extraction failed: {e}")
            return []
    
    def build_graph_from_documents(self, documents: List[Dict]):
        """
        Build knowledge graph from documents
        
        Args:
            documents: List of document chunks with metadata
        """
        logger.info("🔨 Building knowledge graph from documents...")
        start_time = time.time()
        
        entity_counts = Counter()
        
        for doc in documents:
            chunk_text = doc.get('chunk_text', '')
            chunk_id = f"{doc.get('document_id', 'unknown')}_{doc.get('chunk_id', 0)}"
            
            # Extract entities
            entities = self.extract_entities(chunk_text)
            
            for entity_text, entity_type in entities:
                # Normalize entity
                entity_norm = entity_text.lower().strip()
                
                if len(entity_norm) < 2:  # Skip very short entities
                    continue
                
                entity_counts[entity_norm] += 1
                
                # Add entity to graph
                if not self.graph.has_node(entity_norm):
                    self.graph.add_node(entity_norm, type=entity_type, count=1)
                else:
                    self.graph.nodes[entity_norm]['count'] += 1
                
                # Map entity to chunk
                self.entity_to_chunks[entity_norm].add(chunk_id)
        
        # Filter low-frequency entities
        min_freq = settings.KG_MIN_ENTITY_FREQUENCY
        entities_to_remove = [
            entity for entity, count in entity_counts.items()
            if count < min_freq
        ]
        
        for entity in entities_to_remove:
            if entity in self.graph:
                self.graph.remove_node(entity)
            if entity in self.entity_to_chunks:
                del self.entity_to_chunks[entity]
        
        # Create co-occurrence edges
        for doc in documents:
            chunk_text = doc.get('chunk_text', '')
            entities = [e[0].lower().strip() for e in self.extract_entities(chunk_text)]
            entities = [e for e in entities if e in self.graph]
            
            # Add edges between co-occurring entities
            for i, e1 in enumerate(entities):
                for e2 in entities[i+1:]:
                    if self.graph.has_edge(e1, e2):
                        self.graph[e1][e2]['weight'] += 1
                    else:
                        self.graph.add_edge(e1, e2, weight=1)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Knowledge graph built: {self.graph.number_of_nodes()} entities, "
                   f"{self.graph.number_of_edges()} relationships in {elapsed:.2f}s")
    
    def get_related_entities(self, entity: str, max_hops: int = 2) -> Set[str]:
        """
        Get entities related to a given entity via graph traversal
        
        Args:
            entity: Source entity
            max_hops: Maximum number of hops in the graph
            
        Returns:
            Set of related entities
        """
        entity_norm = entity.lower().strip()
        
        if entity_norm not in self.graph:
            return set()
        
        related = set([entity_norm])
        
        try:
            # BFS traversal
            current_level = {entity_norm}
            for _ in range(max_hops):
                next_level = set()
                for node in current_level:
                    if node in self.graph:
                        neighbors = set(self.graph.neighbors(node))
                        next_level.update(neighbors)
                related.update(next_level)
                current_level = next_level
                
                if not current_level:
                    break
        except Exception as e:
            logger.error(f"❌ Graph traversal failed: {e}")
        
        return related
    
    def retrieve_with_kg(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Retrieve documents using knowledge graph + semantic search
        
        Args:
            query: User query
            top_k: Number of results
            
        Returns:
            List of retrieved documents
        """
        start_time = time.time()
        
        try:
            # Extract entities from query
            query_entities = self.extract_entities(query)
            query_entity_texts = [e[0].lower().strip() for e in query_entities]
            
            logger.info(f"🔍 Query entities: {query_entity_texts}")
            
            # Find related entities via graph
            all_related = set()
            for entity in query_entity_texts:
                related = self.get_related_entities(entity, max_hops=2)
                all_related.update(related)
            
            logger.info(f"📊 Found {len(all_related)} related entities")
            
            # Get chunks associated with these entities
            candidate_chunks = set()
            for entity in all_related:
                if entity in self.entity_to_chunks:
                    candidate_chunks.update(self.entity_to_chunks[entity])
            
            # Also do semantic search
            query_embedding = self.embedding_generator.generate_single_embedding(query)
            k = top_k or self.top_k
            semantic_results = vector_db.search(query_embedding, top_k=k * 2)
            
            # Combine results
            combined_results = []
            seen_chunks = set()
            
            # First add KG-based results
            for result in semantic_results:
                chunk_id = f"{result.get('document_id', 'unknown')}_{result.get('chunk_id', 0)}"
                if chunk_id in candidate_chunks and chunk_id not in seen_chunks:
                    result['retrieval_method'] = 'kg+semantic'
                    combined_results.append(result)
                    seen_chunks.add(chunk_id)
            
            # Then add remaining semantic results
            for result in semantic_results:
                chunk_id = f"{result.get('document_id', 'unknown')}_{result.get('chunk_id', 0)}"
                if chunk_id not in seen_chunks and result.get('similarity', 0) >= self.similarity_threshold:
                    result['retrieval_method'] = 'semantic'
                    combined_results.append(result)
                    seen_chunks.add(chunk_id)
            
            # Limit to top_k
            combined_results = combined_results[:k]
            
            retrieval_time = time.time() - start_time
            
            observability.log_retrieval(
                query=query,
                results_count=len(combined_results),
                rag_type="Knowledge Graph RAG",
                retrieval_time=retrieval_time
            )
            
            logger.info(f"🔍 Retrieved {len(combined_results)} documents in {retrieval_time:.2f}s")
            
            return combined_results
            
        except Exception as e:
            logger.error(f"❌ KG retrieval failed: {e}")
            return []
    
    def format_context(self, retrieved_docs: List[Dict]) -> str:
        """Format retrieved documents into context"""
        if not retrieved_docs:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            filename = doc.get('filename', 'Unknown')
            chunk_text = doc.get('chunk_text', doc.get('text', ''))
            similarity = doc.get('similarity', 0)
            method = doc.get('retrieval_method', 'unknown')
            
            context_parts.append(
                f"[Document {i}] {filename} (relevance: {similarity:.2f}, method: {method})\n{chunk_text}"
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
        Complete KG-RAG pipeline
        
        Args:
            query: User query
            top_k: Number of documents
            temperature: LLM temperature
            max_tokens: Maximum tokens
            return_sources: Return source documents
            
        Returns:
            Response dictionary
        """
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Processing KG query: {query}")
            
            # Retrieve with KG
            retrieved_docs = self.retrieve_with_kg(query, top_k=top_k)
            
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
                        "similarity": doc.get('similarity', 0),
                        "method": doc.get('retrieval_method', 'unknown'),
                        "text": doc.get('chunk_text', '')[:200] + "..."
                    }
                    for doc in retrieved_docs
                ]
                response["sources"] = sources
            
            logger.info(f"✅ KG query completed in {total_time:.2f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ KG query failed: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "retrieved_count": 0,
                "total_time": time.time() - start_time
            }

# Test function
def test_kg_rag():
    """Test Knowledge Graph RAG"""
    print("\n" + "="*70)
    print("🕸️  TESTING KNOWLEDGE GRAPH RAG")
    print("="*70 + "\n")
    
    # Check documents
    print("1️⃣  Checking for documents...")
    stats = vector_db.get_stats()
    if stats['total_vectors'] == 0:
        print("   ⚠️  No documents found!")
        return
    print(f"   ✅ Found {stats['total_vectors']} vectors\n")
    
    # Initialize KG RAG
    print("2️⃣  Initializing Knowledge Graph RAG...")
    try:
        kg_rag = KnowledgeGraphRAG(
            provider_name="groq",
            model_name="llama-3.1-8b-instant",
            top_k=3
        )
        print("   ✅ KG RAG initialized\n")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return
    
    # Build knowledge graph
    print("3️⃣  Building knowledge graph...")
    query_embedding = kg_rag.embedding_generator.generate_single_embedding("test")
    all_docs = vector_db.search(query_embedding, top_k=100)
    kg_rag.build_graph_from_documents(all_docs)
    print()
    
    # Test query
    print("4️⃣  Testing KG query...")
    query = "What is this document about?"
    print(f"   Query: {query}\n")
    
    response = kg_rag.query(query, top_k=3, max_tokens=200)
    
    print(f"   📊 Retrieved: {response['retrieved_count']} documents")
    print(f"   ⏱️  Time: {response['total_time']:.2f}s")
    print(f"   💬 Answer: {response['answer'][:200]}...")
    
    if response.get('sources'):
        print(f"\n   📚 Sources:")
        for i, src in enumerate(response['sources'][:2], 1):
            print(f"      {i}. {src['filename']} (method: {src['method']})")
    
    print("\n" + "="*70)
    print("✅ Knowledge Graph RAG tests complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_kg_rag()