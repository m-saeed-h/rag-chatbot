import os  # Import the os module
from dotenv import load_dotenv  # Load the environment variables from the .env file
from langchain_community.document_loaders import PyPDFLoader  # Load the PDF file using the PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter  # Split the documents into chunks
from langchain_openai import OpenAIEmbeddings  # Create embeddings using the OpenAIEmbeddings
from langchain_pinecone import Pinecone  # Push the documents to Pinecone

load_dotenv()  # Load the environment variables from the .env file

def process_pdf(pdf_path):
    """
    Process a PDF file: load, chunk, embed, and store in Pinecone
    
    Args:
        pdf_path (str): Path to the PDF file to process
    """
    try:
        # Load PDF
        loader = PyPDFLoader(pdf_path)  # Load the PDF file using the PyPDFLoader
        documents = loader.load()  # Load the documents from the PDF file
        print(f"Loaded {len(documents)} pages from PDF")

        # Split into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  # Split the documents into chunks
        texts = text_splitter.split_documents(documents)  # Split the documents into chunks
        print(f"Created {len(texts)} chunks")  # Print the number of chunks

        # Create embeddings
        embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))  # Create embeddings using the OpenAIEmbeddings

        # Push to Pinecone  
        Pinecone.from_documents(texts, embeddings, index_name=os.environ.get("INDEX_NAME"))  # Push the documents to Pinecone
        print("Successfully uploaded documents to Pinecone")
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise e

# For backward compatibility, process the default PDF if this script is run directly
if __name__ == "__main__":
    pdf_path = "data/impact_of_generativeAI.pdf"  # Path to the PDF file
    process_pdf(pdf_path)
