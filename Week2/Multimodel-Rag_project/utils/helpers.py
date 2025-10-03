"""
Helper utility functions
Step 2: General utilities
"""
import hashlib
import uuid
from datetime import datetime
from typing import List, Dict, Any
import tiktoken

def generate_document_id(filename: str) -> str:
    """
    Generate unique document ID from filename and timestamp
    
    Args:
        filename: Name of the file
        
    Returns:
        Unique document ID
    """
    timestamp = datetime.now().isoformat()
    unique_string = f"{filename}_{timestamp}"
    return hashlib.md5(unique_string.encode()).hexdigest()

def generate_session_id() -> str:
    """
    Generate unique session ID
    
    Returns:
        UUID string
    """
    return str(uuid.uuid4())

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count tokens in text using tiktoken
    
    Args:
        text: Text to count tokens for
        model: Model name for tokenizer
        
    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback: rough estimation (1 token ≈ 4 characters)
        return len(text) // 4

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
    
    return chunks

def format_sources(sources: List[Dict[str, Any]]) -> str:
    """
    Format source citations for display
    
    Args:
        sources: List of source documents
        
    Returns:
        Formatted citation string
    """
    if not sources:
        return ""
    
    citations = []
    for i, source in enumerate(sources, 1):
        filename = source.get('filename', 'Unknown')
        page = source.get('page', 'N/A')
        citations.append(f"[{i}] {filename} (Page {page})")
    
    return "\n".join(citations)

def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple text similarity (Jaccard similarity)
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score (0-1)
    """
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)

def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """
    Extract top keywords from text (simple frequency-based)
    
    Args:
        text: Input text
        top_n: Number of keywords to extract
        
    Returns:
        List of keywords
    """
    # Simple word frequency
    words = text.lower().split()
    
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 
                  'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
                  'would', 'should', 'could', 'may', 'might', 'can'}
    
    words = [w for w in words if w not in stop_words and len(w) > 3]
    
    # Count frequency
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    return [word for word, freq in sorted_words[:top_n]]

def format_time_ago(timestamp: datetime) -> str:
    """
    Format timestamp as 'time ago'
    
    Args:
        timestamp: Datetime object
        
    Returns:
        Formatted string (e.g., "2 hours ago")
    """
    now = datetime.now()
    diff = now - timestamp
    
    seconds = diff.total_seconds()
    
    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    else:
        days = int(seconds / 86400)
        return f"{days} day{'s' if days > 1 else ''} ago"

# Test function
def test_helpers():
    """Test helper functions"""
    print("Testing Helper Functions...\n")
    
    # Test document ID generation
    doc_id = generate_document_id("test.pdf")
    print(f"✅ Document ID: {doc_id}")
    
    # Test session ID generation
    session_id = generate_session_id()
    print(f"✅ Session ID: {session_id}")
    
    # Test token counting
    text = "This is a test sentence for token counting."
    tokens = count_tokens(text)
    print(f"✅ Token count: {tokens}")
    
    # Test file size formatting
    size = format_file_size(1536000)
    print(f"✅ File size: {size}")
    
    # Test text truncation
    long_text = "This is a very long text that needs to be truncated for display purposes."
    truncated = truncate_text(long_text, 30)
    print(f"✅ Truncated: {truncated}")
    
    # Test text chunking
    chunks = chunk_text("A" * 1500, chunk_size=500, overlap=100)
    print(f"✅ Chunks created: {len(chunks)}")
    
    # Test keyword extraction
    sample_text = "Machine learning and artificial intelligence are transforming the world of technology and data science."
    keywords = extract_keywords(sample_text)
    print(f"✅ Keywords: {keywords}")
    
    # Test time ago formatting
    past_time = datetime(2024, 1, 1, 12, 0, 0)
    time_ago = format_time_ago(past_time)
    print(f"✅ Time ago: {time_ago}")
    
    print("\n✅ All helper functions tested successfully!")

if __name__ == "__main__":
    test_helpers()