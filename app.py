app_code = """
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

st.set_page_config(page_title="Zyro Dynamics HR Portal", page_icon="🚀", layout="centered")
st.title("🚀 Zyro Dynamics HR Help Desk")
st.markdown("Welcome to the internal HR assistant. Ask any question regarding company policies, benefits, guidelines, or leave procedures.")

if os.path.exists("./policies"):
    CORPUS_PATH = "./policies/"
else:
    CORPUS_PATH = "/kaggle/input/zyro-dynamics-hr-corpus/"

@st.cache_resource
def initialize_rag_system():
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 25})
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=512)
    
    oos_prompt = ChatPromptTemplate.from_template(
        "You are a security router for the HR help desk at the company (referred to as Acrux Dynamics or Zyro Dynamics).\\n"
        "Classify if the question is IN_SCOPE (HR policies, leaves, payroll, insurance, PIP, APR, WFH) or OUT_OF_SCOPE (recruitment, ESOP, revenues, products like AcruxCRM, other companies like Zoho).\\n"
        "Respond with exactly one word: 'IN_SCOPE' or 'OUT_OF_SCOPE'.\\n\\nQuestion: {question}\\nClassification:"
    )
    
    rag_prompt = ChatPromptTemplate.from_template(
        "You are a helpful HR Assistant.\\n"
        "Answer the employee's question strictly using the provided context below.\\n"
        "If the context does not contain enough information, answer as much as possible with available facts and clearly state what is missing rather than hallucinating.\\n\\n"
        "Context:\\n{context}\\n\\nQuestion: {question}\\n\\nAnswer:"
    )
    
    return oos_prompt, rag_prompt, llm, retriever

def ask_bot(question, oos_prompt, rag_prompt, llm, retriever):
    classifier = oos_prompt | llm | StrOutputParser()
    decision = classifier.invoke({"question": question}).strip().upper()
    
    if "OUT_OF_SCOPE" in decision:
        return "I can only answer HR-related questions from Zyro Dynamics policy documents.", []
        
    retrieved_docs = retriever.invoke(question)
    context_str = "\\n\\n".join(doc.page_content for doc in retrieved_docs)
    
    rag_chain = rag_prompt | llm | StrOutputParser()
    answer = rag_chain.invoke({"context": context_str, "question": question})
    
    return answer, retrieved_docs

try:
    oos_prompt, rag_prompt, llm, retriever = initialize_rag_system()
except Exception as e:
    st.error(f"Failed to initialize RAG pipeline. Error: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Viewed Sources"):
                for src in message["sources"]:
                    st.caption(f"📍 {src}")

if user_query := st.chat_input("Ask about leaves, travel reimbursements, WFH rules..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        answer, docs = ask_bot(user_query, oos_prompt, rag_prompt, llm, retriever)
        response_placeholder.markdown(answer)
        
        sources_list = []
        if docs:
            sources_list = sorted(list(set([os.path.basename(doc.metadata.get('source', 'Unknown Policy')) for doc in docs])))
            with st.expander("📚 Viewed Sources"):
                for src in sources_list:
                    st.caption(f"📍 {src}")
                    
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources_list})
"""

with open("app.py", "w") as f:
    f.write(app_code.strip())

print("app.py updated successfully.")