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

CORPUS_PATH = "./policies/"  # Ensure this path contains the PDF files with HR policies

@st.cache_resource
def initialize_rag_system():
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)  # increased overlap for dense policy tables
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Synced with notebook Cell 8: k=7, fetch_k=30
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 7, "fetch_k": 30}
    )

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=1024)

    # Synced with notebook Cell 11
    oos_prompt = ChatPromptTemplate.from_template(
        """You are a security router for the HR help desk at Zyro Dynamics (also known as Acrux Dynamics — treat these as the same company).

Classify the question as IN_SCOPE or OUT_OF_SCOPE.

IN_SCOPE — HR/company policy topics:
- Leave policies: Earned Leave, Sick Leave, Maternity Leave (accrual, carry-forward, eligibility)
- Payroll: salary credit dates, payroll cut-off dates
- Compensation: CTC ranges, bonus targets for internal grades (e.g. L4 Senior)
- Health insurance: coverage details, who is covered, premiums
- PIP (Performance Improvement Plan): triggers, duration, conditions
- APR (Annual Performance Review): timelines, self-assessment, increment/promotion letters
- WFH/hybrid work: eligibility, rules, types of arrangements

OUT_OF_SCOPE — everything else:
- Job applications, recruitment, hiring processes
- ESOP, stock options, vesting schedules for new joiners
- Company financials: revenue, profits, performance
- Product features (AcruxCRM, Salesforce comparisons)
- Other companies' policies (Zoho, Freshworks, etc.)
- General knowledge, coding, math, jokes, world news

IMPORTANT: Questions asking about Acrux Dynamics HR policies are IN_SCOPE — treat Acrux Dynamics = Zyro Dynamics.

Respond with EXACTLY one word — either IN_SCOPE or OUT_OF_SCOPE.

Question: {question}
Classification:"""
    )

    # Synced with notebook Cell 10
    rag_prompt = ChatPromptTemplate.from_template(
        """You are an HR Assistant for this company. Answer the employee's question using ONLY the provided context below.

ABSOLUTE RULES — violation means failure:
1. COMPANY NAME: The company may appear as "Acrux Dynamics" or "Zyro Dynamics" in questions or context. These are THE SAME COMPANY. Never acknowledge, comment on, or mention any difference. Never write phrases like "I must correct", "the context refers to", "also referred to as", "not Acrux Dynamics", "not Zyro Dynamics". Simply answer as if there is one company.
2. NO META-PHRASES: Never start with "According to", "Based on the context", "The context states", "Based on the provided". Jump directly to the answer.
3. IF NOT IN CONTEXT: Say "This information is not available in the current HR policy documents." — nothing more.
4. PRECISION: Use exact numbers, dates, ranges, and timelines from the context.

Context:
{context}

Question: {question}

Answer:"""
    )

    return oos_prompt, rag_prompt, llm, retriever


def ask_bot(question, oos_prompt, rag_prompt, llm, retriever):
    classifier = oos_prompt | llm | StrOutputParser()
    decision = classifier.invoke({"question": question}).strip().upper()

    if "OUT_OF_SCOPE" in decision:
        return "I can only answer HR-related questions from Zyro Dynamics policy documents.", []

    retrieved_docs = retriever.invoke(question)
    context_str = "\n\n".join(doc.page_content for doc in retrieved_docs)

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
            sources_list = sorted(list(set([
                os.path.basename(doc.metadata.get('source', 'Unknown Policy'))
                for doc in docs
            ])))
            with st.expander("📚 Viewed Sources"):
                for src in sources_list:
                    st.caption(f"📍 {src}")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources_list})