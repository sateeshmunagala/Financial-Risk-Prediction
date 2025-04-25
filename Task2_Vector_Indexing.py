
import Task1_DataIngestion_Preprocessing as data_ingestion

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
from langchain.text_splitter import TokenTextSplitter

def create_vector_index(text_corpus):
    """Creates a FAISS vector index from the preprocessed text corpus."""
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    documents = [f"Content: {text}" for text in text_corpus.values()]
    metadatas = [{"company": company} for company in text_corpus.keys()]
    docsearch = FAISS.from_texts(documents, embeddings, metadatas=metadatas)
    return docsearch

# --- Example Usage Below---
text_directory = 'data/text_data'

if __name__ == "__main__":
    text_corpus = data_ingestion.load_and_preprocess_text_data(text_directory)

    vector_index = create_vector_index(text_corpus)
    print("\nFAISS vector index created.")