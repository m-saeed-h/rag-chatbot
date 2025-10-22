import streamlit as st
import os
import tempfile
from statelessbot import get_rag_chain
from ingestion import process_pdf


st.title("RAG Chatbot - Talk to your PDF!")
st.write("Upload a PDF document and ask questions about it!")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    # Process the PDF and create/update vector store
    if st.button("Process PDF"):
        with st.spinner("Processing PDF and creating embeddings..."):
            try:
                process_pdf(tmp_file_path)
                st.success("PDF processed successfully! You can now ask questions.")
                st.session_state.pdf_processed = True
                st.session_state.pdf_name = uploaded_file.name
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
    
    # Clean up temporary file
    os.unlink(tmp_file_path)

# Question input and answer generation
if st.session_state.get('pdf_processed', False):
    st.write(f"**Current Document:** {st.session_state.get('pdf_name', 'Unknown')}")
    
    question = st.text_input("Ask a question about the document:")
    
    if st.button("Get answer"):
        if question:
            with st.spinner("Thinking..."):
                try:
                    rag_chain = get_rag_chain()
                    answer = rag_chain.invoke(question)
                    st.write("**Answer**")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
        else:
            st.warning("Please Enter a Question First!")
else:
    st.info("Please upload and process a PDF file first.")

