

import os
from langchain_core.documents import Document
import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = "chroma_store"
COLLECTION = "ticket_collection"
DB_PATH = os.path.join("data", "tickets.db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_ticket_data(db_file):
    import sqlite3
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tickets where status='resolved'")
    rows = cursor.fetchall()
    docs = []
    for row in rows:
        content = (
            f"Issue: {row['issue_type']}\n"
            f"Description: {row['description']}\n"
            f"Resolution: {row['resolution']}"
        )
        docs.append(Document(
            page_content=content, metadata={"source": "tickets.db", "ticket_id": row["ticket_id"], "category": row["category"], "status": row["status"]}
            ))
    conn.close()
    return docs

def create_chroma_collection(chroma_dir, collection_name):
   print(f"Creating Chroma collection '{collection_name}' in directory '{chroma_dir}'...")
   docs = load_ticket_data(DB_PATH)
   print(f"Loaded {len(docs)} tickets from '{DB_PATH}'")

   print('Initializing embedding model...')
   embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

   print('Creating Chroma client...')
   vector_store = Chroma.from_documents(
       documents=docs,
       embedding=embeddings,
       collection_name=collection_name,
       persist_directory=chroma_dir)
   print(f"Collection '{collection_name}' created with {len(docs)} documents.")

if __name__ == "__main__":
    create_chroma_collection(CHROMA_DIR, COLLECTION)