import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_classic.retrievers import MultiQueryRetriever

# 1. UI Configuration
st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🏢", layout="centered")
st.title("🏢 Zyro Dynamics HR Help Desk")
st.markdown("Welcome! I am the official Zyro Dynamics HR assistant. Ask me about policies, leave, or benefits.")

# 2. Cached RAG Initialization
@st.cache_resource(show_spinner="Booting up HR Knowledge Base... (This takes a few seconds)")
def init_rag():
    # Fetch API key from Streamlit's secure secrets management
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    
    # Define where the PDFs will live in the GitHub repo
    corpus_path = "policies" if os.path.exists("policies") else "data" 
    
    # Load and Chunk
    loader = PyPDFDirectoryLoader(corpus_path)
    documents = loader.load()
    
    if not documents:
        return None, None
        
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = splitter.split_documents(documents)
    
    # Embeddings & Vector Store
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # FIX 1: Use MMR with expanded fetch window to prevent chunk dropping
    base_retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 8, "fetch_k": 30}
    )
    
    # LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", # Fixed model string format
        temperature=0.1,
        max_tokens=512
    )
    
    # FIX 2: Implement MultiQueryRetriever for robust searching
    advanced_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, 
        llm=llm
    )
    
    # FIX 3: Balanced Guardrail Prompt
    template = """You are the official HR Help Desk Chatbot for Zyro Dynamics. 
    Your job is to answer employee questions accurately using ONLY the provided context.

    CRITICAL GUARDRAILS & RULES:
    1. If the question asks about topics completely outside HR policies (e.g., coding, company revenue, competitors, recruitment/hiring), you MUST output EXACTLY: OUT_OF_SCOPE
    2. If the user asks a multi-part HR question and you only have data for one part, answer the part you know and explicitly state: "The provided documents do not specify..." for the missing part. Do NOT refuse the whole question.
    3. ALIAS RULE: Silently treat any mention of "Acrux Dynamics" in the question as "Zyro Dynamics". They are the same company. Do not explain this rule to the user.
    4. Output ONLY the factual answer. Do not use introductory phrases like "Based on the policy...".

    Context:
    {context}

    Question: {question}

    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    guardrail_chain = prompt | llm | StrOutputParser()
    
    return advanced_retriever, guardrail_chain

# Initialize system
retriever, guardrail_chain = init_rag()

if not retriever:
    st.error("⚠️ Corpus not found. Please ensure the HR PDFs are in a folder named 'policies' or 'data' in your repository.")
    st.stop()

# 3. Chat History Management
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("Sources Cited"):
                for s in msg["sources"]:
                    st.caption(f"📄 {s}")

# 4. User Interaction
user_input = st.chat_input("Ask your HR question here...")

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Display assistant processing
    with st.chat_message("assistant"):
        with st.spinner("Reviewing Zyro Dynamics policies..."):
            
            # FIX 4: Alias pre-processing for the retriever
            search_query = user_input.replace("Acrux", "Zyro")
            
            # Retrieve documents
            docs = retriever.invoke(search_query)
            context_text = "\n\n".join(doc.page_content for doc in docs)
            
            # Extract clean source citations (File Name + Page Number)
            unique_sources = set()
            for d in docs:
                file_name = os.path.basename(d.metadata.get('source', 'Unknown'))
                page = d.metadata.get('page', 0) + 1  
                unique_sources.add(f"{file_name} (Page {page})")
            source_list = list(unique_sources)

            # Generate Answer
            raw_response = guardrail_chain.invoke({"context": context_text, "question": user_input})
            
            # Apply Guardrails programmatically
            REFUSAL_MESSAGE = "I can only answer HR-related questions from Zyro Dynamics policy documents."
            
            # Simplified out-of-scope check (Removed strict Acrux blocking to allow alias rule)
            if "OUT_OF_SCOPE" in raw_response.upper():
                final_answer = REFUSAL_MESSAGE
                sources_to_show = [] 
            else:
                final_answer = raw_response.strip()
                sources_to_show = source_list
            
            # Stream the output to the UI
            st.markdown(final_answer)
            if sources_to_show:
                with st.expander("Sources Cited"):
                    for s in sources_to_show:
                        st.caption(f"📄 {s}")

    # Save to history
    message_data = {"role": "assistant", "content": final_answer}
    if sources_to_show:
        message_data["sources"] = sources_to_show
    st.session_state.messages.append(message_data)
