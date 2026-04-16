import httpx
import logging
from typing import Literal

from config.settings import settings

# Set up logging for this module
logger = logging.getLogger(__name__)

# Type alias for allowed status values
DocumentStatus = Literal["INDEXED", "FAILED"]


class CallbackService:
    """
    Service responsible for notifying Spring Boot about document processing status.
    
    This service makes PATCH requests to the Spring Boot backend to update
    the document status in PostgreSQL.
    """
    
    def __init__(self):
        """Initialize the callback service with Spring Boot URL and service key."""
        self.spring_boot_url = settings.SPRING_BOOT_URL.rstrip("/")
        self.service_key = settings.INTERNAL_SERVICE_KEY
        self.timeout_seconds = 10  # Fail fast if Spring Boot is unresponsive
    
    def _build_url(self, document_id: int) -> str:
        """
        Construct the full callback URL for a specific document.
        
        Args:
            document_id: The database ID of the document
            
        Returns:
            Full URL like: http://localhost:8080/api/documents/123/status
        """
        return f"{self.spring_boot_url}/api/documents/{document_id}/status"
    
    def _build_headers(self) -> dict:
        """
        Build HTTP headers including the service-to-service authentication key.
        
        Returns:
            Dictionary of HTTP headers
        """
        return {
            "Content-Type": "application/json",
            "X-Service-Key": self.service_key
        }
    
    def update_document_status(
        self, 
        document_id: int, 
        status: DocumentStatus
    ) -> bool:
        """
        Notify Spring Boot to update the document status in PostgreSQL.
        
        This method makes a synchronous PATCH request. It is designed to fail
        gracefully—errors are logged but not raised, so the main indexing
        operation can still return success.
        
        Args:
            document_id: The database ID of the document
            status: Either "INDEXED" or "FAILED"
            
        Returns:
            True if the callback succeeded, False otherwise
        """
        url = self._build_url(document_id)
        headers = self._build_headers()
        payload = {"status": status}
        
        logger.info(f"Updating document {document_id} status to '{status}' via {url}")
        
        try:
            # Use a context manager to ensure the connection is properly closed
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.patch(
                    url,
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()  # Raises exception for 4xx/5xx status codes
                
                logger.info(f"Successfully updated document {document_id} status to '{status}'")
                return True
                
        except httpx.TimeoutException:
            logger.error(
                f"Timeout while updating document {document_id} status. "
                f"Spring Boot at {self.spring_boot_url} did not respond within {self.timeout_seconds}s."
            )
            return False
            
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error {e.response.status_code} while updating document {document_id} status: "
                f"{e.response.text}"
            )
            return False
            
        except httpx.RequestError as e:
            logger.error(
                f"Connection error while updating document {document_id} status. "
                f"Cannot reach Spring Boot at {self.spring_boot_url}: {str(e)}"
            )
            return False
            
        except Exception as e:
            logger.exception(
                f"Unexpected error while updating document {document_id} status: {str(e)}"
            )
            return False


# Create a singleton instance for use throughout the application
callback_service = CallbackService()