

import os
from langchain_core.documents import Document
import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = "chroma_store"
COLLECTION = "faq_collection"
CSV_PATH = os.path.join("data", "faq.csv")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_faq_data(csv_file):
    df = pd.read_csv(csv_file)
    docs = []
    for _, row in df.iterrows():
        content = f"Question: {row['question']}\nAnswer: {row['answer']}"
        docs.append(Document(
            page_content=content, metadata={"source": "faq.csv", "category": row.get('category', 'general')}
            ))
    return docs

def create_chroma_collection(chroma_dir, collection_name):
   print(f"Creating Chroma collection '{collection_name}' in directory '{chroma_dir}'...")
   docs = load_faq_data(CSV_PATH)
   print(f"Loaded {len(docs)} FAQ entries from '{CSV_PATH}'")

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