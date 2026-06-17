
import os
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = "chroma_store"
COLLECTION = "guide_collection"
PDF_PATH = os.path.join("data", "telecom_guide.pdf")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


def load_and_split_pdf(pdf_path: str) -> list[Document]:
    """Load PDF and split into chunks using RecursiveCharacterTextSplitter."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    return chunks


def ingest_pdf_to_vector_store(pdf_path: str) -> Chroma:
    """Load PDF, split into chunks, embed, and store in Chroma vector database."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    print(f"Loading PDF from {pdf_path}...")
    chunks = load_and_split_pdf(pdf_path)
    print(f"Split PDF into {len(chunks)} chunks")

    print(f"Creating embeddings using {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Storing chunks in Chroma vector database at {CHROMA_DIR}...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR
    )

    print(f"Successfully ingested {len(chunks)} chunks into vector store")
    return vector_store


def get_vector_store() -> Chroma:
    """Retrieve existing Chroma vector store or create new one."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

    return vector_store


if __name__ == "__main__":
    vector_store = ingest_pdf_to_vector_store(PDF_PATH)
    print("\nVector store ready for querying!")
