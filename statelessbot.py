import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- Load Pinecone Vector Store ---
embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])
vectorstore = Pinecone(
    index_name=os.environ["INDEX_NAME"],
    embedding=embeddings
)

# --- Create Chat Model ---
chat = ChatOpenAI(temperature=0, model="gpt-4o-mini")

# --- Create Retriever ---
retriever = vectorstore.as_retriever()

# --- Create Prompt Template ---
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# --- Create RAG Chain using LCEL (LangChain Expression Language) ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
{"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | chat
    | StrOutputParser()
)

# # --- Ask Questions ---
# res1 = rag_chain.invoke("What is this document even about?")
# print(res1)

# res2 = rag_chain.invoke("How does it concern me?")
# print(res2)