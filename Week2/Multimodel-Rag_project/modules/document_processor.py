"""
Document Processing Module
Step 5: Document Upload, Processing, Chunking, and Embedding
"""
import os
import io
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time
import numpy as np

# Document processing
import PyPDF2
from docx import Document as DocxDocument
from PIL import Image
import pytesseract

# Embeddings
from sentence_transformers import SentenceTransformer

from config.settings import settings
from utils.logger import setup_logger
from utils.helpers import generate_document_id, chunk_text, format_file_size
from utils.database import vector_db, memory_db
from utils.observability import observability

logger = setup_logger("document_processor")

# ==================== Text Extractors ====================

class TextExtractor:
    """Extract text from various document formats"""
    
    @staticmethod
    def extract_from_txt(file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            logger.info(f"✅ Extracted {len(text)} characters from TXT")
            return text
        except Exception as e:
            logger.error(f"❌ Failed to extract from TXT: {e}")
            raise
    
    @staticmethod
    def extract_from_pdf(file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                num_pages = len(pdf_reader.pages)
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
            
            logger.info(f"✅ Extracted {len(text)} characters from PDF ({num_pages} pages)")
            return text
        except Exception as e:
            logger.error(f"❌ Failed to extract from PDF: {e}")
            raise
    
    @staticmethod
    def extract_from_docx(file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = DocxDocument(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
            logger.info(f"✅ Extracted {len(text)} characters from DOCX")
            return text
        except Exception as e:
            logger.error(f"❌ Failed to extract from DOCX: {e}")
            raise
    
    @staticmethod
    def extract_from_image(file_path: str) -> str:
        """Extract text from image using OCR"""
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            
            logger.info(f"✅ Extracted {len(text)} characters from image via OCR")
            return text
        except Exception as e:
            logger.error(f"❌ Failed to extract from image: {e}")
            # Return empty string if OCR fails (Tesseract might not be installed)
            logger.warning("⚠️ OCR failed - Tesseract might not be installed")
            return ""

# ==================== Embedding Generator ====================

class EmbeddingGenerator:
    """Generate embeddings using sentence-transformers"""
    
    def __init__(self, model_name: str = None):
        """
        Initialize embedding generator
        
        Args:
            model_name: Name of the sentence-transformers model
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            logger.info(f"📥 Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"✅ Embedding model loaded (dimension: {self.model.get_sentence_embedding_dimension()})")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        try:
            start_time = time.time()
            embeddings = self.model.encode(texts, show_progress_bar=False)
            elapsed = time.time() - start_time
            
            logger.info(f"✅ Generated {len(embeddings)} embeddings in {elapsed:.2f}s")
            return embeddings
        except Exception as e:
            logger.error(f"❌ Failed to generate embeddings: {e}")
            raise
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        return self.generate_embeddings([text])[0]

# ==================== Document Processor ====================

class DocumentProcessor:
    """Main document processing class"""
    
    def __init__(self):
        self.text_extractor = TextExtractor()
        self.embedding_generator = EmbeddingGenerator()
    
    def process_document(
        self,
        file_path: str,
        filename: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> Dict:
        """
        Process a document: extract text, chunk, generate embeddings, and store
        
        Args:
            file_path: Path to the document file
            filename: Original filename
            chunk_size: Size of text chunks (default from settings)
            chunk_overlap: Overlap between chunks (default from settings)
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        try:
            # Get file info
            file_size = os.path.getsize(file_path)
            file_ext = Path(filename).suffix.lower()
            document_id = generate_document_id(filename)
            
            logger.info(f"📄 Processing document: {filename} ({format_file_size(file_size)})")
            
            # Extract text based on file type
            if file_ext == '.txt':
                text = self.text_extractor.extract_from_txt(file_path)
            elif file_ext == '.pdf':
                text = self.text_extractor.extract_from_pdf(file_path)
            elif file_ext == '.docx':
                text = self.text_extractor.extract_from_docx(file_path)
            elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                text = self.text_extractor.extract_from_image(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            if not text or len(text.strip()) < 10:
                raise ValueError("Extracted text is too short or empty")
            
            # Chunk the text
            chunk_size = chunk_size or settings.CHUNK_SIZE
            chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
            
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
            logger.info(f"📝 Created {len(chunks)} chunks")
            
            # Generate embeddings
            embeddings = self.embedding_generator.generate_embeddings(chunks)
            
            # Prepare metadata for each chunk
            metadata = []
            for i, chunk in enumerate(chunks):
                meta = {
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_id": i,
                    "chunk_text": chunk,
                    "file_type": file_ext,
                    "total_chunks": len(chunks)
                }
                metadata.append(meta)
            
            # Store in vector database
            vector_db.add_documents(embeddings, metadata)
            
            # Store document metadata in SQL database
            memory_db.add_document_metadata(
                document_id=document_id,
                filename=filename,
                file_type=file_ext,
                file_size=file_size,
                chunk_count=len(chunks)
            )
            
            # Update status
            memory_db.update_document_status(document_id, "indexed")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Log to observability
            observability.log_document_processing(
                filename=filename,
                file_type=file_ext,
                chunks_created=len(chunks),
                processing_time=processing_time
            )
            
            result = {
                "success": True,
                "document_id": document_id,
                "filename": filename,
                "file_size": file_size,
                "chunks": len(chunks),
                "processing_time": processing_time
            }
            
            logger.info(f"✅ Document processed successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def process_uploaded_file(
        self,
        uploaded_file,
        save_to_disk: bool = True
    ) -> Dict:
        """
        Process an uploaded file (from Streamlit file uploader)
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            save_to_disk: Whether to save the file to disk
            
        Returns:
            Processing results
        """
        try:
            filename = uploaded_file.name
            
            # Save file temporarily
            if save_to_disk:
                file_path = settings.DOCUMENTS_PATH / filename
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
            else:
                # Save to temp location
                file_path = f"/tmp/{filename}"
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
            
            # Process the document
            result = self.process_document(file_path, filename)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to process uploaded file: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_all_documents(self) -> List[Dict]:
        """Get all processed documents"""
        try:
            docs = memory_db.get_all_documents()
            return [
                {
                    "document_id": doc.document_id,
                    "filename": doc.filename,
                    "file_type": doc.file_type,
                    "file_size": format_file_size(doc.file_size),
                    "chunk_count": doc.chunk_count,
                    "status": doc.status,
                    "upload_time": doc.upload_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                }
                for doc in docs
            ]
        except Exception as e:
            logger.error(f"❌ Failed to get documents: {e}")
            return []
    
    def delete_document(self, document_id: str):
        """Delete a document and its embeddings"""
        try:
            # Delete from vector database
            vector_db.delete_documents(document_id)
            
            # Delete metadata
            memory_db.delete_document_metadata(document_id)
            
            logger.info(f"🗑️ Deleted document {document_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete document: {e}")
            return False

# Global document processor instance
document_processor = DocumentProcessor()

# ==================== Test Function ====================

def test_document_processor():
    """Test document processing"""
    print("\n" + "="*70)
    print("📄 TESTING DOCUMENT PROCESSOR")
    print("="*70 + "\n")
    
    # Create a test text file
    test_file_path = settings.DOCUMENTS_PATH / "test_document.txt"
    test_content = """
    This is a test document for the RAG system.
    It contains multiple sentences to test chunking.
    The document processor will extract this text,
    split it into chunks, generate embeddings,
    and store them in the vector database.
    This allows us to test the entire pipeline.
    """
    
    with open(test_file_path, 'w') as f:
        f.write(test_content)
    
    print("1️⃣  Processing test document...")
    result = document_processor.process_document(
        str(test_file_path),
        "test_document.txt"
    )
    
    if result['success']:
        print(f"   ✅ Document processed successfully!")
        print(f"   📊 Document ID: {result['document_id']}")
        print(f"   📝 Chunks created: {result['chunks']}")
        print(f"   ⏱️  Processing time: {result['processing_time']:.2f}s")
    else:
        print(f"   ❌ Processing failed: {result.get('error')}")
    
    print("\n2️⃣  Retrieving all documents...")
    docs = document_processor.get_all_documents()
    print(f"   ✅ Found {len(docs)} document(s)")
    for doc in docs:
        print(f"      - {doc['filename']} ({doc['chunk_count']} chunks)")
    
    print("\n3️⃣  Testing embedding search...")
    try:
        query = "test document RAG system"
        query_embedding = document_processor.embedding_generator.generate_single_embedding(query)
        results = vector_db.search(query_embedding, top_k=3)
        print(f"   ✅ Search returned {len(results)} results")
        for i, result in enumerate(results, 1):
            filename = result.get('filename', 'Unknown')
            chunk_text = result.get('chunk_text', result.get('text', 'No text'))
            similarity = result.get('similarity', 0)
            print(f"      {i}. {filename} (similarity: {similarity:.3f})")
            print(f"         Text: {chunk_text[:100]}...")
    except Exception as e:
        print(f"   ❌ Search failed: {e}")
    
    print("\n" + "="*70)
    print("✅ Document processor tests complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_document_processor()