from pydantic import BaseModel


class ProcessDocumentRequest(BaseModel):
    """Request model for the /internal/documents/process endpoint."""
    document_id: int
    file_path: str


class ProcessDocumentResponse(BaseModel):
    """Response model returned after processing completes."""
    document_id: int
    status: str
    chunks_created: int
    message: str