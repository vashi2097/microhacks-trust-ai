"""
RAG Service - retrieves documents from Azure AI Search.
Shared across all models for a fair comparison.
"""
import logging
from dataclasses import dataclass, field
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery, QueryType
from azure.core.credentials import AzureKeyCredential
from .config import Settings

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """You are an HR assistant helping employees with questions about their benefits and HR policies.
Answer ONLY using the facts from the sources below. If there is not enough information, say you don't know.
Do not generate answers that don't use the sources below.
Always cite the source name in square brackets for each fact you use.

{sources}
"""


@dataclass
class Document:
    content: str
    title: str = ""
    source: str = ""


@dataclass
class RetrievalResult:
    documents: list = field(default_factory=list)
    context: str = ""
    numbered_chunks: str = ""


class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.search_client = SearchClient(
            endpoint=settings.azure_ai_search_endpoint,
            index_name=settings.azure_search_index_name,
            credential=AzureKeyCredential(settings.azure_search_api_key),
        )

    def retrieve(self, query: str) -> RetrievalResult:
        try:
            results = self.search_client.search(
                search_text=query,
                query_type=QueryType.SEMANTIC,
                semantic_configuration_name=self.settings.semantic_configuration_name,
                vector_queries=[
                    VectorizableTextQuery(
                        text=query,
                        k_nearest_neighbors=self.settings.search_top_k,
                        fields=self.settings.vector_field_name,
                    )
                ],
                top=self.settings.search_top_k,
                select=["id", "content", "title"],
            )
            documents = []
            for r in results:
                documents.append(Document(
                    content=r.get("content", ""),
                    title=r.get("title", ""),
                    source=r.get("title", ""),
                ))
        except Exception as e:
            logger.error(f"Search error: {e}")
            documents = []

        context_parts = []
        for doc in documents:
            name = doc.source or doc.title or "unknown"
            context_parts.append(f"{name}: {doc.content}")
        context = "\n\n".join(context_parts)

        numbered_chunks = ""
        for i, doc in enumerate(documents, 1):
            name = doc.source or doc.title or "unknown"
            numbered_chunks += f"Chunk {i} [{name}]:\n{doc.content[:400]}\n\n"

        return RetrievalResult(documents=documents, context=context, numbered_chunks=numbered_chunks)

    def system_prompt(self, context: str) -> str:
        return RAG_SYSTEM_PROMPT.format(sources=context)
