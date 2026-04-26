import logging
from typing import List

from langchain_core.prompts import PromptTemplate

from config.llm_config import get_llm
from models.query_models import QueryRequest, QueryResponse, SourceChunk
from services.vector_store_service import vector_store_service

logger = logging.getLogger(__name__)

# ── Prompt Template ────────────────────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an internal enterprise assistant for a Moroccan company.
Answer the employee's question using ONLY the context provided below.
If the answer is not in the context, say clearly that you cannot find
the answer in the available documents.
Always respond in the same language as the question.
Cite the document name for each piece of information you use.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
)

# Cosine distance threshold — above this means "not relevant enough"
# Lowered from 0.8 to 0.6 for higher citation precision (BURAQ-26)
SIMILARITY_THRESHOLD = 0.65


class RAGService:
    """
    Service responsible for the full RAG pipeline:
    Retrieve → Augment → Generate
    """

    def _build_context_string(self, documents: List[str], metadatas: List[dict]) -> str:
        """
        Concatenate retrieved chunks into a single context string
        with source labels for the LLM to cite.
        """
        context_parts = []
        for doc, meta in zip(documents, metadatas):
            filename = meta.get("filename", "unknown")
            page = meta.get("page_number", 0)
            context_parts.append(f"[Source: {filename}, Page: {page}]\n{doc}")
        return "\n\n".join(context_parts)

    def _build_source_chunks(
        self,
        documents: List[str],
        metadatas: List[dict],
        distances: List[float]
    ) -> List[SourceChunk]:
        """
        Convert raw ChromaDB results into SourceChunk Pydantic models.
        Only includes chunks whose similarity distance is below SIMILARITY_THRESHOLD
        (i.e. chunks that were actually relevant to the answer).
        Results are sorted by page number ascending so citations appear in document order.
        """
        sources = []
        for doc, meta, distance in zip(documents, metadatas, distances):
            # Only include high-relevance chunks
            if distance <= SIMILARITY_THRESHOLD:
                sources.append(SourceChunk(
                    document_id=int(meta.get("document_id", 0)),
                    filename=meta.get("filename", "unknown"),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    page_number=int(meta.get("page_number", 0)),
                    # Clean extra whitespace from excerpt before trimming to 200 chars
                    excerpt=" ".join(doc.split())[:200]
                ))

        # Sort by page number so citations appear in document order
        sources.sort(key=lambda s: s.page_number)
        return sources

    def _calculate_confidence(self, distances: List[float]) -> float:
        """
        Convert average cosine distance into a confidence score between 0 and 1.
        Distance 0.0 → confidence 1.0 (perfect match)
        Distance 1.0 → confidence 0.0 (no match)
        """
        if not distances:
            return 0.0
        avg_distance = sum(distances) / len(distances)
        return round(1.0 - avg_distance, 4)

    def query_documents(self, question: str, language: str = "en") -> QueryResponse:
        """
        Run the full RAG pipeline for a user question.

        Args:
            question: The user's natural language question
            language: Language code ("en", "fr", "ar")

        Returns:
            QueryResponse with answer, sources, confidence score,
            language_detected, and should_create_ticket
        """
        logger.info(f"RAG query received | language={language} | question={question[:80]}")

        # ── STEP 1: RETRIEVE ───────────────────────────────────────────────────
        try:
            collection = vector_store_service.get_chroma_collection()
            results = collection.query(
                query_texts=[question],
                n_results=5,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return QueryResponse(
                question=question,
                answer="An error occurred while searching the knowledge base.",
                sources=[],
                has_answer=False,
                confidence_score=0.0,
                language_detected=language,
                should_create_ticket=True
            )

        # Unpack results (ChromaDB returns lists of lists — one per query)
        documents: List[str] = results["documents"][0]
        metadatas: List[dict] = results["metadatas"][0]
        distances: List[float] = results["distances"][0]

        # ── STEP 2: CHECK RELEVANCE ────────────────────────────────────────────
        if not documents or distances[0] > SIMILARITY_THRESHOLD:
            logger.info(f"No relevant chunks found (best distance={distances[0] if distances else 'N/A'})")
            return QueryResponse(
                question=question,
                answer="I could not find an answer to your question in the available documents.",
                sources=[],
                has_answer=False,
                confidence_score=0.0,
                language_detected=language,
                should_create_ticket=True
            )

        # ── STEP 3: AUGMENT ────────────────────────────────────────────────────
        context_string = self._build_context_string(documents, metadatas)
        # Pass distances so _build_source_chunks can filter by relevance threshold
        source_chunks = self._build_source_chunks(documents, metadatas, distances)
        confidence = self._calculate_confidence(distances)

        # ── STEP 4: GENERATE ───────────────────────────────────────────────────
        try:
            llm = get_llm()
            chain = RAG_PROMPT_TEMPLATE | llm
            response = chain.invoke({
                "context": context_string,
                "question": question
            })
            # LangChain chat models return an AIMessage — extract the text
            answer = response.content if hasattr(response, "content") else str(response)
            logger.info("LLM response received successfully")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return QueryResponse(
                question=question,
                answer="The AI model is currently unavailable. Please try again later.",
                sources=source_chunks,
                has_answer=False,
                confidence_score=confidence,
                language_detected=language,
                should_create_ticket=True
            )

        return QueryResponse(
            question=question,
            answer=answer,
            sources=source_chunks,
            has_answer=True,
            confidence_score=confidence,
            language_detected=language,
            should_create_ticket=False
        )


# Singleton instance — same pattern as your other services
rag_service = RAGService()