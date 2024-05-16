import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM as LlamaHuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
import torch

# Setup for caching the index and LLM to avoid reloading
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def setup_llama_index():
    # Define and configure the embedding model
    embed_model = LangchainEmbedding(SentenceTransformer('sentence-transformers/all-mpnet-base-v2'))

    # Define and configure the Llama LLM
    llama_llm = LlamaHuggingFaceLLM(
        context_window=4096,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.0, "do_sample": False},
        system_prompt="You are a Q&A assistant...",
        query_wrapper_prompt=SimpleInputPrompt("{query_str}"),
        tokenizer_name="HuggingFaceH4/zephyr-7b-beta",
        model_name="HuggingFaceH4/zephyr-7b-beta",
        device_map="auto",
        model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True}
    )

    # Load documents and create the index
    documents = SimpleDirectoryReader('/content/data').load_data()  # Assuming document data is in this directory
    service_context = ServiceContext.from_defaults(chunk_size=1024, llm=llama_llm, embed_model=embed_model)
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    return index.as_query_engine()

def extract_text_from_pdf(file):
    """ Extract text from the uploaded PDF file using pdfplumber. """
    text = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  # Ensure that text extraction was successful
                text.append(page_text)
    return " ".join(text)

def main():
    st.title('PDF Reader and Question Answering with RAG-like Model')

    # Load the query engine only once
    query_engine = setup_llama_index()

    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])
    if uploaded_file is not None:
        document_text = extract_text_from_pdf(uploaded_file)
        if document_text:
            st.text_area("Extracted Text", document_text, height=300)
        else:
            st.error("No text could be extracted from the PDF. Please check the file and try again.")

        question = st.text_input("Ask a question based on the PDF")
        if st.button("Get Answer"):
            if question:
                # Simulate RAG-like query using the index and LLM
                response = query_engine.query(question)
                st.text_area("Answer", response, height=150)
            else:
                st.error("Please enter a question to get an answer.")

if __name__ == "__main__":
    main()
