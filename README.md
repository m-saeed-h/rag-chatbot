# RAG Document Chatbot

AI-powered chatbot that answers questions about documents using Retrieval Augmented Generation (RAG).

## 🚀 Features
- Document processing and chunking
- Vector similarity search
- Natural language responses using OpenAI
- Simple web interface

## 🛠️ Tech Stack
- Python
- LangChain
- OpenAI API
- Pinecone/ChromaDB
- Streamlit

## 📦 Installation
```bash
pip install -r requirements.txt
```

## 🔑 Setup

1. Get OpenAI API key from openai.com
2. Create `.env` file:
```
OPENAI_API_KEY=your_key_here
```

## 🚀 Usage
```bash
streamlit run app.py
```

## 📝 How It Works

1. Documents are loaded and split into chunks
2. Chunks are converted to embeddings
3. Embeddings stored in vector database
4. User query is embedded and similar chunks retrieved
5. LLM generates answer using retrieved context

## 🎯 Future Improvements
- [ ] Support more document types
- [ ] Add chat history
- [ ] Improve chunking strategy
- [ ] Add source citations

## 📄 License
MIT
```

---

## .gitignore
```
# API Keys
.env
*.env

# Python
__pycache__/
*.pyc
.ipynb_checkpoints/

# Data
data/
*.pdf
*.txt

# Virtual Environment
venv/
env/
