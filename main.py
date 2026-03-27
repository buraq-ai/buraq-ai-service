from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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