app_code = """
import streamlit as st
import os
import time
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# 1. Page Configuration & Styling
st.set_page_config(page_title="Zyro Dynamics HR Portal", page_icon="🚀", layout="centered")
st.title("🚀 Zyro Dynamics HR Help Desk")
st.markdown("Welcome to the internal HR assistant. Ask any question regarding company policies, benefits, guidelines, or leave procedures.")

CORPUS_PATH = "/kaggle/input/zyro-dynamics-hr-corpus/" # Fallback for local testing, update if necessary for Streamlit Cloud deployment

# 2. Pipeline Initialization (Cached for blazing performance)
@st.cache_resource
def initialize_rag_system():
    # Load
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()
    
    # Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    
    # Embed & Index
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Retriever
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20})
    
    # LLM Setup
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=512)
    
    # Prompts
    oos_prompt = ChatPromptTemplate.from_template(
        "You are a security router for an HR help desk chatbot at Zyro Dynamics.\\n"
        "Classify if the question is an HR or company policy query ('IN_SCOPE') or completely unrelated ('OUT_OF_SCOPE').\\n"
        "Respond with exactly one word: 'IN_SCOPE' or 'OUT_OF_SCOPE'.\\n\\nQuestion: {question}\\nClassification:"
    )
    
    rag_prompt = ChatPromptTemplate.from_template(
        "You are a helpful HR Assistant for Zyro Dynamics.\\n"
        "Answer the employee's question strictly using the provided context below.\\n"
        "If the answer is not contained in the context, politely state that you cannot answer the question based on current HR documents.\\n\\n"
        "Context:\\n{context}\\n\\nQuestion: {question}\\n\\nAnswer:"
    )
    
    return oos_prompt, rag_prompt, llm, retriever

# Guardrail-enabled routing engine
def ask_bot(question, oos_prompt, rag_prompt, llm, retriever):
    # Guardrail Check
    classifier = oos_prompt | llm | StrOutputParser()
    decision = classifier.invoke({"question": question}).strip().upper()
    
    if "OUT_OF_SCOPE" in decision:
        return "I can only answer HR-related questions from Zyro Dynamics policy documents.", []
        
    # RAG Execution
    retrieved_docs = retriever.invoke(question)
    context_str = "\\n\\n".join(doc.page_content for doc in retrieved_docs)
    
    rag_chain = rag_prompt | llm | StrOutputParser()
    answer = rag_chain.invoke({"context": context_str, "question": question})
    
    return answer, retrieved_docs

# Initialize Components
try:
    oos_prompt, rag_prompt, llm, retriever = initialize_rag_system()
except Exception as e:
    st.error(f"Failed to initialize RAG pipeline. Ensure your API keys are set up correctly. Error: {e}")
    st.stop()

# 3. Maintaining Chat State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Viewed Sources"):
                for src in message["sources"]:
                    st.caption(f"📍 {src}")

# 4. Chat Input Handling
if user_query := st.chat_input("Ask about leaves, travel reimbursements, WFH rules..."):
    # Display user comment
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Process and display assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        answer, docs = ask_bot(user_query, oos_prompt, rag_prompt, llm, retriever)
        response_placeholder.markdown(answer)
        
        # Format sources extracted from metadata
        sources_list = []
        if docs:
            sources_list = sorted(list(set([os.path.basename(doc.metadata.get('source', 'Unknown Policy File')) for doc in docs])))
            with st.expander("📚 Viewed Sources"):
                for src in sources_list:
                    st.caption(f"📍 {src}")
                    
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources_list})
"""

with open("app.py", "w") as f:
    f.write(app_code.strip())

print("app.py created successfully.")