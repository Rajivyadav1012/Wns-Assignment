"""
Database Management - Vector Store (FAISS) and Memory (SQLite)
Step 3: Database Setup
"""
import faiss
import numpy as np
import pickle
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger("database")

# ==================== SQLAlchemy Base ====================
Base = declarative_base()

# ==================== Database Models ====================

class ConversationMessage(Base):
    """Table for storing conversation messages"""
    __tablename__ = 'conversation_messages'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    tokens_used = Column(Integer, nullable=True)
    model_used = Column(String(100), nullable=True)

class DocumentMetadata(Base):
    """Table for storing document metadata"""
    __tablename__ = 'document_metadata'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String(100), unique=True, nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(20), nullable=False)
    file_size = Column(Integer, nullable=False)
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    chunk_count = Column(Integer, default=0)
    status = Column(String(20), default="uploaded")

# ==================== FAISS Vector Database ====================

class VectorDatabase:
    """Manage FAISS vector database for document embeddings"""
    
    def __init__(self, dimension: int = 384):
        """
        Initialize FAISS vector database
        
        Args:
            dimension: Embedding dimension (default 384 for all-MiniLM-L6-v2)
        """
        self.dimension = dimension
        self.index = None
        self.document_map = {}  # Maps index position to document metadata
        self.index_path = settings.VECTOR_DB_PATH / "faiss.index"
        self.metadata_path = settings.VECTOR_DB_PATH / "metadata.pkl"
        
        self.initialize()
    
    def initialize(self):
        """Initialize or load existing FAISS index"""
        try:
            if self.index_path.exists() and self.metadata_path.exists():
                # Load existing index
                self.index = faiss.read_index(str(self.index_path))
                with open(self.metadata_path, 'rb') as f:
                    self.document_map = pickle.load(f)
                logger.info(f"✅ Loaded existing FAISS index with {self.index.ntotal} vectors")
            else:
                # Create new index (using L2 distance)
                self.index = faiss.IndexFlatL2(self.dimension)
                logger.info(f"✅ Created new FAISS index (dimension: {self.dimension})")
        except Exception as e:
            logger.error(f"❌ Failed to initialize FAISS: {e}")
            # Create new index as fallback
            self.index = faiss.IndexFlatL2(self.dimension)
    
    def add_documents(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict]
    ):
        """
        Add document embeddings to the index
        
        Args:
            embeddings: Numpy array of embeddings (shape: [n, dimension])
            metadata: List of metadata dicts for each embedding
        """
        try:
            # Ensure embeddings are float32
            embeddings = embeddings.astype('float32')
            
            # Get current index size
            start_idx = self.index.ntotal
            
            # Add to FAISS index
            self.index.add(embeddings)
            
            # Store metadata
            for i, meta in enumerate(metadata):
                self.document_map[start_idx + i] = meta
            
            # Save to disk
            self.save()
            
            logger.info(f"✅ Added {len(embeddings)} embeddings to index")
        except Exception as e:
            logger.error(f"❌ Failed to add documents: {e}")
            raise
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of documents with metadata and similarity scores
        """
        try:
            # Ensure query is float32 and 2D
            query_embedding = query_embedding.astype('float32')
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Search
            distances, indices = self.index.search(query_embedding, top_k)
            
            # Prepare results
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.document_map):
                    continue
                
                result = self.document_map[idx].copy()
                result['distance'] = float(dist)
                result['similarity'] = 1 / (1 + float(dist))  # Convert distance to similarity
                results.append(result)
            
            logger.info(f"🔍 Found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
    
    def delete_documents(self, document_id: str):
        """
        Delete documents by document_id
        Note: FAISS doesn't support direct deletion, so we rebuild the index
        
        Args:
            document_id: Document ID to delete
        """
        try:
            # Find indices to keep
            indices_to_keep = []
            new_document_map = {}
            
            for idx, meta in self.document_map.items():
                if meta.get('document_id') != document_id:
                    indices_to_keep.append(idx)
            
            if len(indices_to_keep) == len(self.document_map):
                logger.warning(f"⚠️ Document {document_id} not found")
                return
            
            # Rebuild index with remaining documents
            if indices_to_keep:
                # Get embeddings for documents to keep
                # This requires storing embeddings separately or rebuilding from source
                logger.warning("⚠️ FAISS deletion requires rebuilding index - feature pending")
            else:
                # All documents deleted, reset index
                self.index = faiss.IndexFlatL2(self.dimension)
                self.document_map = {}
                self.save()
                logger.info(f"✅ Deleted all documents")
                
        except Exception as e:
            logger.error(f"❌ Failed to delete documents: {e}")
    
    def save(self):
        """Save index and metadata to disk"""
        try:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.document_map, f)
            logger.debug("💾 Saved FAISS index and metadata")
        except Exception as e:
            logger.error(f"❌ Failed to save index: {e}")
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "total_documents": len(set(meta.get('document_id') for meta in self.document_map.values()))
        }
    
    def reset(self):
        """Reset the entire index"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.document_map = {}
        self.save()
        logger.info("🗑️ Reset FAISS index")

# ==================== SQLite Memory Database ====================

class MemoryDatabase:
    """Manage SQLite database for conversation memory and metadata"""
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self.initialize()
    
    def initialize(self):
        """Initialize SQLite database and create tables"""
        try:
            db_url = f"sqlite:///{settings.MEMORY_DB_PATH}"
            self.engine = create_engine(
                db_url,
                connect_args={"check_same_thread": False},
                echo=False
            )
            
            # Create all tables
            Base.metadata.create_all(bind=self.engine)
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            logger.info(f"✅ SQLite database initialized at {settings.MEMORY_DB_PATH}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize SQLite: {e}")
            raise
    
    def get_session(self):
        """Get a new database session"""
        return self.SessionLocal()
    
    # ==================== Conversation Methods ====================
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tokens_used: Optional[int] = None,
        model_used: Optional[str] = None
    ):
        """Add a conversation message"""
        session = self.get_session()
        try:
            message = ConversationMessage(
                session_id=session_id,
                role=role,
                content=content,
                tokens_used=tokens_used,
                model_used=model_used
            )
            session.add(message)
            session.commit()
            logger.debug(f"💬 Added {role} message for session {session_id[:8]}")
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Failed to add message: {e}")
            raise
        finally:
            session.close()
    
    def get_recent_messages(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[ConversationMessage]:
        """Get recent messages for a session"""
        session = self.get_session()
        try:
            messages = session.query(ConversationMessage)\
                .filter(ConversationMessage.session_id == session_id)\
                .order_by(ConversationMessage.timestamp.desc())\
                .limit(limit)\
                .all()
            return list(reversed(messages))
        finally:
            session.close()
    
    def clear_session(self, session_id: str):
        """Clear all messages for a session"""
        session = self.get_session()
        try:
            session.query(ConversationMessage)\
                .filter(ConversationMessage.session_id == session_id)\
                .delete()
            session.commit()
            logger.info(f"🗑️ Cleared session {session_id[:8]}")
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Failed to clear session: {e}")
            raise
        finally:
            session.close()
    
    # ==================== Document Metadata Methods ====================
    
    def add_document_metadata(
        self,
        document_id: str,
        filename: str,
        file_type: str,
        file_size: int,
        chunk_count: int = 0
    ):
        """Add document metadata"""
        session = self.get_session()
        try:
            doc = DocumentMetadata(
                document_id=document_id,
                filename=filename,
                file_type=file_type,
                file_size=file_size,
                chunk_count=chunk_count
            )
            session.add(doc)
            session.commit()
            logger.info(f"📄 Added metadata for {filename}")
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Failed to add document metadata: {e}")
            raise
        finally:
            session.close()
    
    def get_all_documents(self) -> List[DocumentMetadata]:
        """Get all document metadata"""
        session = self.get_session()
        try:
            return session.query(DocumentMetadata).all()
        finally:
            session.close()
    
    def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        """Get specific document metadata"""
        session = self.get_session()
        try:
            return session.query(DocumentMetadata)\
                .filter(DocumentMetadata.document_id == document_id)\
                .first()
        finally:
            session.close()
    
    def delete_document_metadata(self, document_id: str):
        """Delete document metadata"""
        session = self.get_session()
        try:
            session.query(DocumentMetadata)\
                .filter(DocumentMetadata.document_id == document_id)\
                .delete()
            session.commit()
            logger.info(f"🗑️ Deleted metadata for {document_id[:8]}")
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Failed to delete document metadata: {e}")
            raise
        finally:
            session.close()
    
    def update_document_status(self, document_id: str, status: str):
        """Update document processing status"""
        session = self.get_session()
        try:
            doc = session.query(DocumentMetadata)\
                .filter(DocumentMetadata.document_id == document_id)\
                .first()
            if doc:
                doc.status = status
                session.commit()
        finally:
            session.close()

# ==================== Global Database Instances ====================

vector_db = VectorDatabase(dimension=settings.EMBEDDING_DIMENSION)
memory_db = MemoryDatabase()

logger.info("✅ All databases initialized successfully")

# ==================== Test Function ====================

def test_databases():
    """Test database functionality"""
    print("\n" + "="*70)
    print("🔍 TESTING DATABASES")
    print("="*70 + "\n")
    
    # Test Vector DB
    print("1️⃣  Testing Vector Database (FAISS)...")
    try:
        # Create dummy embeddings
        dummy_embeddings = np.random.rand(3, settings.EMBEDDING_DIMENSION).astype('float32')
        metadata = [
            {"document_id": "doc1", "chunk_id": 0, "text": "Test chunk 1"},
            {"document_id": "doc1", "chunk_id": 1, "text": "Test chunk 2"},
            {"document_id": "doc2", "chunk_id": 0, "text": "Test chunk 3"}
        ]
        
        vector_db.add_documents(dummy_embeddings, metadata)
        print(f"   ✅ Added {len(dummy_embeddings)} test vectors")
        
        # Test search
        query = np.random.rand(settings.EMBEDDING_DIMENSION).astype('float32')
        results = vector_db.search(query, top_k=2)
        print(f"   ✅ Search returned {len(results)} results")
        
        # Get stats
        stats = vector_db.get_stats()
        print(f"   ✅ Stats: {stats['total_vectors']} vectors, {stats['total_documents']} documents")
        
    except Exception as e:
        print(f"   ❌ Vector DB test failed: {e}")
    
    # Test Memory DB
    print("\n2️⃣  Testing Memory Database (SQLite)...")
    try:
        # Test conversation
        session_id = "test_session"
        memory_db.add_message(session_id, "user", "Hello!")
        memory_db.add_message(session_id, "assistant", "Hi there!")
        print("   ✅ Added test messages")
        
        messages = memory_db.get_recent_messages(session_id, limit=10)
        print(f"   ✅ Retrieved {len(messages)} messages")
        
        # Test document metadata
        memory_db.add_document_metadata(
            document_id="test_doc",
            filename="test.pdf",
            file_type="pdf",
            file_size=1024,
            chunk_count=3
        )
        print("   ✅ Added test document metadata")
        
        docs = memory_db.get_all_documents()
        print(f"   ✅ Retrieved {len(docs)} documents")
        
    except Exception as e:
        print(f"   ❌ Memory DB test failed: {e}")
    
    print("\n" + "="*70)
    print("✅ Database tests complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_databases()