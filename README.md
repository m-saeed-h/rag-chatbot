# RAG Chatbot

AI-powered chatbot that answers questions about documents using Retrieval Augmented Generation. Now supports uploading any PDF document!

## Features
- **Dynamic PDF Upload**: Upload any PDF document through the web interface
- Document processing and embedding
- Vector similarity search using Pinecone
- Context-aware responses using OpenAI GPT-4o-mini
- Real-time processing with progress indicators

## Tech Stack
- Python
- LangChain
- OpenAI API
- Pinecone Vector Database
- Streamlit (Web Interface)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
INDEX_NAME=your_pinecone_index_name_here
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## How It Works

### Step 1: PDF Upload & Processing
- Upload any PDF document through the web interface
- The PDF is processed: loaded, split into chunks, embedded using OpenAI embeddings, and stored in Pinecone vector database

### Step 2: Question Processing
- User provides a query through the web interface

### Step 3: Vector Search
- The query is converted into an embedding
- Pinecone compares it with stored embeddings to find the most semantically similar chunks
- These chunks become the context for the LLM

### Step 4: Response Generation
- The retrieved chunks are formatted into a single string and plugged into a prompt template:
```
Answer the question based only on the following context:
{context}

Question: {question}
```

### Step 5: Answer Delivery
- The prompt is sent to the OpenAI chat model (GPT-4o-mini) to generate a response
- The answer is displayed in the web interface

## Usage

1. **Upload PDF**: Use the file uploader to select any PDF document
2. **Process Document**: Click "Process PDF" to create embeddings and store in Pinecone
3. **Ask Questions**: Type your questions about the document content
4. **Get Answers**: Receive context-aware answers based on the uploaded document

## File Structure

- `app.py`: Main Streamlit application with file upload and chat interface
- `ingestion.py`: PDF processing functions (load, chunk, embed, store)
- `statelessbot.py`: RAG chain creation and question answering logic
- `requirements.txt`: Python dependencies
- `data/`: Directory for storing PDF files (optional, now supports dynamic uploads)

## Next Steps
- [x] Add Streamlit UI with file upload
- [x] Support dynamic PDF uploads
- [ ] Add support for multiple document types (DOCX, TXT, etc.)
- [ ] Implement document management (view uploaded documents)
- [ ] Add conversation history
- [ ] Deploy to cloud
- [ ] Improve chunking strategy
