import logging
from fastapi import APIRouter, HTTPException, status

from models.query_models import QueryRequest, QueryResponse
from services.rag_service import rag_service

# Set up logger for this module
logger = logging.getLogger(__name__)

# Create the router with a prefix and tags for OpenAPI documentation
router = APIRouter(
    prefix="/api",
    tags=["Query - RAG Pipeline"],
)


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question using RAG",
    description="""
    Core RAG query endpoint called by Spring Boot on behalf of Angular users.

    This endpoint:
    1. Embeds the user's question using all-MiniLM-L6-v2
    2. Retrieves the top 5 most relevant chunks from ChromaDB
    3. Checks similarity — if no relevant chunks found, returns has_answer=false
    4. Builds an augmented prompt with retrieved context
    5. Calls the configured LLM (Ollama or OpenAI)
    6. Returns the answer with source citations and confidence score
    """
)
def query_documents(request: QueryRequest) -> QueryResponse:
    """
    Run the RAG pipeline for a user question.

    Args:
        request: Contains question (str) and language (str, default "en")

    Returns:
        QueryResponse with answer, sources, has_answer, and confidence_score

    Raises:
        HTTPException: If an unexpected error occurs (500)
    """
    logger.info(
        f"Query request received | language={request.language} | "
        f"question={request.question[:80]}"
    )

    try:
        response = rag_service.query_documents(
            question=request.question,
            language=request.language
        )

        logger.info(
            f"Query completed | has_answer={response.has_answer} | "
            f"confidence={response.confidence_score} | "
            f"sources={len(response.sources)}"
        )

        return response

    except Exception as e:
        error_msg = f"Unexpected error during RAG query: {str(e)}"
        logger.exception(error_msg)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )