import os
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentService:
    """
    Service responsible for loading documents from disk and splitting them into chunks.
    """
    
    # Supported file extensions and their corresponding LangChain loader classes
    SUPPORTED_EXTENSIONS = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
    }
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the document service with chunking configuration.
        
        Args:
            chunk_size: Maximum number of characters per chunk
            chunk_overlap: Number of overlapping characters between adjacent chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]  # Default separators (paragraph → line → space → char)
        )
    
    def _get_file_extension(self, file_path: str) -> str:
        """
        Extract the lowercase file extension from a path.
        
        Args:
            file_path: Full path to the file
            
        Returns:
            Lowercase file extension including the dot (e.g., '.pdf')
        """
        return os.path.splitext(file_path)[1].lower()
    
    def _load_document(self, file_path: str) -> List[Document]:
        """
        Load a document using the appropriate LangChain loader based on file extension.
        
        Args:
            file_path: Full path to the document file
            
        Returns:
            List of LangChain Document objects (usually one Document containing all pages)
            
        Raises:
            ValueError: If the file extension is not supported
            FileNotFoundError: If the file does not exist at the given path
        """
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document file not found: {file_path}")
        
        # Determine file extension
        extension = self._get_file_extension(file_path)
        
        # Get the appropriate loader class
        loader_class = self.SUPPORTED_EXTENSIONS.get(extension)
        if loader_class is None:
            supported = ", ".join(self.SUPPORTED_EXTENSIONS.keys())
            raise ValueError(
                f"Unsupported file extension '{extension}'. "
                f"Supported extensions: {supported}"
            )
        
        # Instantiate loader and load the document
        loader = loader_class(file_path)
        documents = loader.load()
        
        return documents
    
    def _extract_filename(self, file_path: str) -> str:
        """
        Extract just the filename from a full path.
        
        Args:
            file_path: Full path to the file
            
        Returns:
            Filename with extension (e.g., 'contract.pdf')
        """
        return os.path.basename(file_path)
    
    def process_document(
        self, 
        document_id: int, 
        file_path: str
    ) -> Tuple[List[Document], str]:
        """
        Load a document from disk and split it into chunks.
        
        Args:
            document_id: The database ID of the document (for metadata)
            file_path: Full path to the document file on disk
            
        Returns:
            Tuple containing:
                - List of chunked LangChain Document objects with enriched metadata
                - The extracted filename string
                
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file type is unsupported
            Exception: For other loading/splitting errors
        """
        # Extract filename for metadata
        filename = self._extract_filename(file_path)
        
        # Load the raw document
        raw_documents = self._load_document(file_path)
        
        # Split the document into chunks
        chunks = self.text_splitter.split_documents(raw_documents)
        
        # Enrich each chunk with additional metadata
        for idx, chunk in enumerate(chunks):
            # Preserve existing metadata from the loader (like page_number for PDFs)
            # and add our custom metadata fields
            chunk.metadata.update({
                "document_id": document_id,
                "filename": filename,
                "chunk_index": idx,
            })
            # For PDFs, 'page' is often in metadata as 'page' or 'page_number'
            # We standardize it to 'page_number' for consistency
            if "page" in chunk.metadata:
                chunk.metadata["page_number"] = chunk.metadata["page"]
            elif "page_number" not in chunk.metadata:
                # Default to 0 if no page info is available
                chunk.metadata["page_number"] = 0
        
        return chunks, filename


# Create a singleton instance for use throughout the application
document_service = DocumentService()