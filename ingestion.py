import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone

load_dotenv()

# Load PDF
loader = PyPDFLoader("data/impact_of_generativeAI.pdf")
documents = loader.load()

# Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
print(f"created {len(texts)} chunks")

# Create embeddings
embeddings = OpenAIEmbeddings(openai_api_type=os.environ.get("OPENAI_API_KEY"))

# Push to Pinecone
Pinecone.from_documents(
    texts,
    embeddings,
    index_name=os.environ.get("INDEX_NAME")
)
