from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def load_retriever():
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    retriever = FAISS.load_local("models/faiss_index", embedding).as_retriever()
    return retriever
