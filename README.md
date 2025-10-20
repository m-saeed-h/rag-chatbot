# RAG Chatbot

AI-powered chatbot that answers questions about documents using Retrieval Augmented Generation.

## Features
- Document processing and embedding
- Vector similarity search
- Context-aware responses using OpenAI

## Tech Stack
- Python
- LangChain
- OpenAI API
- Pinecone

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Add your API keys to `.env`:
```
OPENAI_API_KEY=your_key_here
```

3. Run:
```bash
python ingestion.py
python statelessbot.py
```

## How It Works
Step 1
PDF Documents are ingested into RAG pipeline
(Loaded, thereafter splitted into small chunks, then embedded into vectors using openAI vectors and then stored in Pinecone vectorised Database) 

Step 2
User provides a query

step 3
The query is converted into an embedding.
Pinecone compares it with stored embeddings to find the most semantically similar chunks.
These chunks become the context for the LLM.

step 4
The retrieved chunks are formatted into a single string (using format_docs) and plugged into a prompt template:
"""
Answer the question based only on the following context:
{context}

Question: {question}
"""

step 5
The prompt is sent to the OpenAI chat model (ChatOpenAI) to generate a response.
The output is parsed into a plain string (StrOutputParser) for easy use.

## Next Steps
- [ ] Add Streamlit UI
- [ ] Deploy to cloud
- [ ] Improve chunking strategy
