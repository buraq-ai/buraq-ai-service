import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
from langchain_core.documents import Document

from config.settings import settings


class VectorStoreService:
    """
    Service responsible for generating embeddings and storing document chunks in ChromaDB.
    """
    
    def __init__(self):
        """
        Initialize the vector store service with ChromaDB HTTP client connection.
        """
        self.host = settings.CHROMA_HOST
        self.port = settings.CHROMA_PORT
        self.collection_name = settings.CHROMA_COLLECTION_NAME
        
        # Initialize the embedding function (runs locally, no API calls)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Connect to ChromaDB via HTTP (Docker container)
        self.client = chromadb.HttpClient(
            host=self.host,
            port=self.port
        )
    
    def get_chroma_collection(self):
        """
        Get or create the ChromaDB collection for document chunks.
        
        If the collection doesn't exist, it will be created with the
        sentence-transformer embedding function.
        
        Returns:
            ChromaDB Collection object
        """
        try:
            # Try to get existing collection
            collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
        except Exception:
            # Collection doesn't exist, create it
            collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity for semantic search
            )
        
        return collection
    
    def store_document_chunks(
        self, 
        chunks: List[Document], 
        document_id: int, 
        filename: str
    ) -> int:
        """
        Store document chunks in ChromaDB with embeddings and metadata.
        
        Args:
            chunks: List of LangChain Document objects (already chunked and metadata-enriched)
            document_id: The database ID of the document
            filename: Original filename (for metadata)
            
        Returns:
            Number of chunks successfully stored
            
        Raises:
            Exception: If storage fails (connection error, etc.)
        """
        # Get or create the collection
        collection = self.get_chroma_collection()
        
        # Prepare data structures for batch insertion
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        
        for chunk in chunks:
            # Generate a unique ID for each chunk
            chunk_id = f"{document_id}_chunk_{chunk.metadata.get('chunk_index', 0)}"
            
            # Extract the text content
            text_content = chunk.page_content
            
            # Prepare metadata (ensure all values are JSON-serializable)
            chunk_metadata = {
                "document_id": document_id,
                "filename": filename,
                "chunk_index": chunk.metadata.get("chunk_index", 0),
                "page_number": chunk.metadata.get("page_number", 0),
            }
            
            # Add any additional metadata from the loader (like PDF page numbers)
            # but only include simple types that ChromaDB can handle
            for key, value in chunk.metadata.items():
                if key not in chunk_metadata and isinstance(value, (str, int, float, bool)):
                    chunk_metadata[key] = value
            
            ids.append(chunk_id)
            documents.append(text_content)
            metadatas.append(chunk_metadata)
        
        # Batch insert all chunks
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        return len(ids)
    
    def verify_document_exists(self, document_id: int) -> bool:
        """
        Check if a document has already been indexed in ChromaDB.
        
        Args:
            document_id: The database ID of the document
            
        Returns:
            True if at least one chunk exists for this document_id, False otherwise
        """
        try:
            collection = self.get_chroma_collection()
            # Query for chunks with this document_id
            results = collection.get(
                where={"document_id": document_id},
                limit=1  # We only need to know if any exist
            )
            return len(results.get("ids", [])) > 0
        except Exception:
            return False


# Create a singleton instance for use throughout the application
vector_store_service = VectorStoreService()