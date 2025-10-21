import streamlit as st
from statelessbot import rag_chain


st.title("RAG Chatbot")
question = st.text_input("Ask a question about the document:")

if st.button("Get answer"):
    if question:
        with st.spinner("Thinking..."):
            answer = rag_chain.invoke(question)

        st.write("**Answer**")
        st.write(answer)
    
    else:
        st.warning("Please Enter a Question First!")

