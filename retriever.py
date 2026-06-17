from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = "chroma_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Collection names
FAQ_COLLECTION = "faq_collection"
TICKET_COLLECTION = "ticket_collection"
GUIDE_COLLECTION = "guide_collection"


class VectorRetriever:
    """Retrieve documents from Chroma vector store for different ingestion sources."""

    def __init__(self, chroma_dir: str = CHROMA_DIR, embedding_model: str = EMBEDDING_MODEL):
        """Initialize the retriever with embeddings."""
        self.chroma_dir = chroma_dir
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self._vector_stores = {}

    def _get_vector_store(self, collection_name: str) -> Chroma:
        """Get or create a vector store for the specified collection."""
        if collection_name not in self._vector_stores:
            self._vector_stores[collection_name] = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.chroma_dir
            )
        return self._vector_stores[collection_name]

    def retrieve_from_faq(self, query: str, k: int = 5) -> list[dict]:
        """Retrieve relevant FAQ documents."""
        vector_store = self._get_vector_store(FAQ_COLLECTION)
        results = vector_store.similarity_search(query, k=k)

        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": "faq"
            }
            for doc in results
        ]

    def retrieve_from_tickets(self, query: str, k: int = 5) -> list[dict]:
        """Retrieve relevant ticket documents."""
        vector_store = self._get_vector_store(TICKET_COLLECTION)
        results = vector_store.similarity_search(query, k=k)

        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": "tickets"
            }
            for doc in results
        ]

    def retrieve_from_guide(self, query: str, k: int = 5) -> list[dict]:
        """Retrieve relevant guide documents."""
        vector_store = self._get_vector_store(GUIDE_COLLECTION)
        results = vector_store.similarity_search(query, k=k)

        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": "guide"
            }
            for doc in results
        ]

    def retrieve_all(self, query: str, k: int = 5) -> dict:
        """Retrieve relevant documents from all collections."""
        results = {
            "faq": self.retrieve_from_faq(query, k),
            "tickets": self.retrieve_from_tickets(query, k),
            "guide": self.retrieve_from_guide(query, k)
        }
        return results

    def retrieve_with_scores(self, query: str, k: int = 5, source: str = None) -> list[dict]:
        """
        Retrieve documents with similarity scores.

        Args:
            query: Search query
            k: Number of results to return
            source: Specific source to search ("faq", "tickets", "guide", or None for all)
        """
        if source == "faq":
            vector_store = self._get_vector_store(FAQ_COLLECTION)
        elif source == "tickets":
            vector_store = self._get_vector_store(TICKET_COLLECTION)
        elif source == "guide":
            vector_store = self._get_vector_store(GUIDE_COLLECTION)
        else:
            raise ValueError(f"Invalid source: {source}. Must be 'faq', 'tickets', or 'guide'")

        results = vector_store.similarity_search_with_relevance_scores(query, k=k)

        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": source,
                "relevance_score": score
            }
            for doc, score in results
        ]


def get_retriever() -> VectorRetriever:
    """Get a retriever instance."""
    return VectorRetriever()


if __name__ == "__main__":
    # Example usage
    retriever = get_retriever()

    # Test query
    test_query = "What are the latest plans?"

    print(f"Searching for: '{test_query}'\n")

    # Retrieve from specific sources
    print("=== FAQ Results ===")
    faq_results = retriever.retrieve_from_faq(test_query, k=3)
    for result in faq_results:
        print(f"\n{result['content']}")
        print(f"Category: {result['metadata'].get('category', 'N/A')}")

    print("\n\n=== Ticket Results ===")
    ticket_results = retriever.retrieve_from_tickets(test_query, k=3)
    for result in ticket_results:
        print(f"\n{result['content']}")
        print(f"Ticket ID: {result['metadata'].get('ticket_id', 'N/A')}")

    print("\n\n=== Guide Results ===")
    guide_results = retriever.retrieve_from_guide(test_query, k=3)
    for result in guide_results:
        print(f"\n{result['content']}")
        print(f"Page: {result['metadata'].get('page', 'N/A')}")

    # Retrieve with scores
    print("\n\n=== Results with Relevance Scores (FAQ) ===")
    scored_results = retriever.retrieve_with_scores(test_query, k=3, source="faq")
    for result in scored_results:
        print(f"\nScore: {result['relevance_score']:.4f}")
        print(f"{result['content'][:100]}...")
