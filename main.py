import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# NEW: Import the document router
from routers.document_router import router as document_router

# Configure logging for the entire application
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(
    title="Buraq-AI Service",
    description="AI-powered RAG microservice for Buraq-AI helpdesk",
    version="0.1.0"
)

# CORS configuration (allows Angular frontend and Java backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://localhost:8081"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# NEW: Register the document processing router
app.include_router(document_router)

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "UP",
        "service": "buraq-ai-service"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)