from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd

# Load data
df = pd.read_csv('data/shl_catalog.csv')

# Create combined text entries
texts = df.apply(
    lambda row: f"{row['Product Name']}: {row['Description']} "
                f"(Type: {row['Assessment Type']}, Skills: {row['Skills Measured']})",
    axis=1
).tolist()

# Embedding model
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Vector store
db = FAISS.from_texts(texts, embedding)

# Save index
db.save_local("models/faiss_index")
