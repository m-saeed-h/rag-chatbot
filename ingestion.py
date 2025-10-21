import os  # Import the os module
from dotenv import load_dotenv  # Load the environment variables from the .env file
from langchain_community.document_loaders import PyPDFLoader  # Load the PDF file using the PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter  # Split the documents into chunks
from langchain_openai import OpenAIEmbeddings  # Create embeddings using the OpenAIEmbeddings
from langchain_pinecone import Pinecone  # Push the documents to Pinecone

load_dotenv()  # Load the environment variables from the .env file

# Load PDF
pdf_path = "data/impact_of_generativeAI.pdf"  # Path to the PDF file
loader = PyPDFLoader(pdf_path)  # Load the PDF file using the PyPDFLoader
documents = loader.load()  # Load the documents from the PDF file

# Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  # Split the documents into chunks
texts = text_splitter.split_documents(documents)  # Split the documents into chunks
print(f"created {len(texts)} chunks")  # Print the number of chunks

# Create embeddings
embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))  # Create embeddings using the OpenAIEmbeddings

# Push to Pinecone  
Pinecone.from_documents(texts, embeddings, index_name=os.environ.get("INDEX_NAME"))  # Push the documents to Pinecone
