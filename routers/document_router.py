import logging
from fastapi import APIRouter, HTTPException, status

from models.document_models import ProcessDocumentRequest, ProcessDocumentResponse
from services.document_service import document_service
from services.vector_store_service import vector_store_service
from services.callback_service import callback_service

# Set up logger for this module
logger = logging.getLogger(__name__)

# Create the router with a prefix and tags for OpenAPI documentation
router = APIRouter(
    prefix="/internal/documents",
    tags=["Internal - Document Processing"],
)


@router.post(
    "/process",
    response_model=ProcessDocumentResponse,
    status_code=status.HTTP_200_OK,
    summary="Process and index a document",
    description="""
    Internal endpoint called by Spring Boot after a document is uploaded.
    
    This endpoint:
    1. Loads the document from disk using the appropriate loader (PDF or DOCX)
    2. Splits it into chunks of ~500 characters with 50-character overlap
    3. Generates embeddings using sentence-transformers (all-MiniLM-L6-v2)
    4. Stores chunks and embeddings in ChromaDB with metadata
    5. Notifies Spring Boot to update the document status to INDEXED or FAILED
    
    This is a synchronous operation—the response is returned after all steps complete.
    """
)
def process_document(request: ProcessDocumentRequest) -> ProcessDocumentResponse:
    """
    Process a document for RAG indexing.
    
    Args:
        request: Contains document_id and file_path
        
    Returns:
        ProcessDocumentResponse with status, chunks_created, and message
        
    Raises:
        HTTPException: If processing fails (500) or file is invalid (400)
    """
    document_id = request.document_id
    file_path = request.file_path
    
    logger.info(f"Processing document {document_id} from path: {file_path}")
    
    try:
        # Step 1: Load and chunk the document
        logger.info(f"Loading and chunking document {document_id}...")
        chunks, filename = document_service.process_document(document_id, file_path)
        logger.info(f"Document {document_id} split into {len(chunks)} chunks")
        
        # Step 2: Store chunks with embeddings in ChromaDB
        logger.info(f"Storing {len(chunks)} chunks in ChromaDB for document {document_id}...")
        chunks_stored = vector_store_service.store_document_chunks(
            chunks=chunks,
            document_id=document_id,
            filename=filename
        )
        logger.info(f"Successfully stored {chunks_stored} chunks for document {document_id}")
        
        # Step 3: Notify Spring Boot of success
        logger.info(f"Notifying Spring Boot of successful indexing for document {document_id}")
        callback_success = callback_service.update_document_status(
            document_id=document_id,
            status="INDEXED"
        )
        
        if not callback_success:
            # Document is indexed but status update failed—log but don't fail the request
            logger.warning(
                f"Document {document_id} indexed successfully but status callback to Spring Boot failed. "
                f"Status may need manual update."
            )
        
        # Step 4: Return success response
        return ProcessDocumentResponse(
            document_id=document_id,
            status="INDEXED",
            chunks_created=chunks_stored,
            message=f"Successfully indexed {chunks_stored} chunks from {filename}"
        )
        
    except FileNotFoundError as e:
        error_msg = f"Document file not found: {file_path}"
        logger.error(f"Document {document_id}: {error_msg}")
        
        # Notify Spring Boot of failure
        callback_service.update_document_status(document_id, "FAILED")
        
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )
        
    except ValueError as e:
        # Typically unsupported file extension
        error_msg = str(e)
        logger.error(f"Document {document_id}: {error_msg}")
        
        # Notify Spring Boot of failure
        callback_service.update_document_status(document_id, "FAILED")
        
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )
        
    except Exception as e:
        # Catch-all for unexpected errors (ChromaDB connection, embedding failure, etc.)
        error_msg = f"Unexpected error processing document: {str(e)}"
        logger.exception(f"Document {document_id}: {error_msg}")
        
        # Notify Spring Boot of failure
        callback_service.update_document_status(document_id, "FAILED")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )