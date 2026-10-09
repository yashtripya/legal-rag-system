"""
Streamlit User Interface for Indian Legal RAG System.
Provides a modern web app for legal document ingestion, Q&A, evidence inspection, and collection monitoring.
"""

import os
import streamlit as st
from pathlib import Path

from config import DEFAULT_TOP_K, DEFAULT_SCORE_THRESHOLD, DEFAULT_LLM_MODEL, SUPPORTED_LLM_MODELS, SAMPLE_DATA_DIR, RAW_DATA_DIR
from src.rag_pipeline import LegalRAGPipeline


# Streamlit Page Configuration
st.set_page_config(
    page_title="Indian Legal RAG System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern visual aesthetics
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .status-badge-online {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-badge-offline {
        background-color: #FEF08A;
        color: #854D0E;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .citation-card {
        background-color: #F8FAFC;
        border-left: 4px solid #2563EB;
        padding: 0.8rem 1rem;
        border-radius: 0.375rem;
        margin-bottom: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline():
    """Singleton pipeline instance cached for Streamlit session."""
    return LegalRAGPipeline()


pipeline = get_pipeline()

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/color/96/scales.png", width=64)
    st.title("System Control")

    # System Status Panel
    sys_status = pipeline.get_system_status()
    vec_stats = sys_status["vector_store"]
    llm_health = sys_status["llm_service"]

    st.markdown("### 📊 Database & Model Status")
    st.write(f"**Indexed Documents:** {vec_stats.get('total_documents', 0)}")
    st.write(f"**Total Chunks:** {vec_stats.get('total_chunks', 0)}")

    if llm_health.get("status") == "online":
        st.markdown('<span class="status-badge-online">🟢 Ollama LLM Online</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge-offline">🟡 Fallback Mode (Retrieval Active)</span>', unsafe_allow_html=True)

    st.divider()

    # RAG Hyperparameters
    st.markdown("### ⚙️ Retrieval Settings")
    top_k = st.slider("Retrieval Depth (top_k)", min_value=1, max_value=10, value=DEFAULT_TOP_K)
    score_threshold = st.slider("Min Similarity Score", min_value=0.0, max_value=0.8, value=DEFAULT_SCORE_THRESHOLD, step=0.05)
    selected_model = st.selectbox("Local LLM Model", options=SUPPORTED_LLM_MODELS, index=0)

    st.divider()

    # Ingestion Actions
    st.markdown("### 📥 Legal Document Ingestion")

    if st.button("Ingest Sample Legal Dataset", type="primary", use_container_width=True):
        with st.spinner("Ingesting Supreme Court judgments & IPC statutes..."):
            summary = pipeline.ingest_directory(SAMPLE_DATA_DIR)
            st.success(f"Ingested {summary.get('successful_files')} sample documents ({summary.get('vector_store_stats', {}).get('total_chunks')} chunks total)!")
            st.rerun()

    uploaded_files = st.file_uploader(
        "Upload Custom Legal PDF/TXT",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload authentic Indian legal documents to extend vector search.",
    )

    if uploaded_files and st.button("Process & Index Uploaded Files", use_container_width=True):
        success_count = 0
        with st.spinner("Processing uploaded legal documents..."):
            for uploaded_file in uploaded_files:
                save_path = RAW_DATA_DIR / uploaded_file.name
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                res = pipeline.ingest_file(save_path)
                if res.get("status") == "success":
                    success_count += 1
                else:
                    st.error(f"Failed to ingest '{uploaded_file.name}': {res.get('error')}")

        if success_count > 0:
            st.success(f"Successfully processed and indexed {success_count} file(s)!")
            st.rerun()

    if st.button("Clear Vector Database", type="secondary", use_container_width=True):
        pipeline.clear_index()
        st.warning("Vector collection reset!")
        st.rerun()


# Main Application Interface
st.markdown('<div class="main-header">⚖️ Indian Legal RAG System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">AI Legal Research Assistant Grounded Exclusively in Indian Supreme Court Judgments, IPC Statutes & Constitutional Provisions</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["💬 Legal Q&A Assistant", "📚 Indexed Legal Library", "🏗️ Architecture & Specs"])

# -----------------------------------------------------------------------------
# TAB 1: LEGAL Q&A ASSISTANT
# -----------------------------------------------------------------------------
with tab1:
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Display Chat Messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "citations" in msg and msg["citations"]:
                with st.expander("🔍 View Verified Legal Source Passages & Citations"):
                    for cit in msg["citations"]:
                        st.markdown(
                            f"**[{cit['citation_id']}] {cit['title']}** ({cit['authority']})\n"
                            f"- Source: `{cit['filename']}` (Page {cit['page']}) | Match: {cit['relevance_score']*100:.1f}%\n"
                            f"> \"{cit['excerpt']}\""
                        )

    # Prompt Suggestions
    if not st.session_state.messages:
        st.markdown("#### 💡 Suggested Legal Queries for Quick Demo:")
        sugg_cols = st.columns(3)
        if sugg_cols[0].button("What is the Basic Structure Doctrine?"):
            prompt_input = "What is the Basic Structure Doctrine established by the Supreme Court of India?"
        elif sugg_cols[1].button("Explain IPC Section 300 Exceptions"):
            prompt_input = "What are the exceptions where culpable homicide is not murder under Section 300 of the IPC?"
        elif sugg_cols[2].button("Is Right to Privacy a Fundamental Right?"):
            prompt_input = "Is the Right to Privacy a fundamental right under Article 21 according to Puttaswamy?"
        else:
            prompt_input = None
    else:
        prompt_input = None

    # Chat Input
    user_query = st.chat_input("Ask a legal question (e.g., 'What is Article 21 protection?')...") or prompt_input

    if user_query:
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate Assistant Answer
        with st.chat_message("assistant"):
            with st.spinner("Searching vector database & synthesizing grounded legal answer..."):
                response = pipeline.query(
                    user_query=user_query,
                    history=st.session_state.messages[:-1],
                    top_k=top_k,
                    score_threshold=score_threshold,
                    model_override=selected_model,
                )

                answer_text = response["answer"]
                citations = response.get("citations", [])

                st.markdown(answer_text)

                if citations:
                    with st.expander("🔍 View Verified Legal Source Passages & Citations"):
                        for cit in citations:
                            st.markdown(
                                f"**[{cit['citation_id']}] {cit['title']}** ({cit['authority']})\n"
                                f"- Source: `{cit['filename']}` (Page {cit['page']}) | Match: {cit['relevance_score']*100:.1f}%\n"
                                f"> \"{cit['excerpt']}\""
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "citations": citations,
                })


# -----------------------------------------------------------------------------
# TAB 2: INDEXED LEGAL LIBRARY
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("📚 Currently Indexed Legal Documents")
    stats = pipeline.vector_store.get_stats()
    
    st.metric("Total Indexed Chunks", stats.get("total_chunks", 0))
    st.metric("Total Document Files", stats.get("total_documents", 0))

    if stats.get("source_files"):
        st.markdown("### Source Document List")
        for fn in stats.get("source_files", []):
            st.write(f"- 📄 `{fn}`")
    else:
        st.info("No legal documents indexed yet. Click 'Ingest Sample Legal Dataset' in the sidebar to get started.")


# -----------------------------------------------------------------------------
# TAB 3: SYSTEM ARCHITECTURE & INTERVIEW SPECIFICATIONS
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("🏗️ Indian Legal RAG System Architecture")
    
    st.markdown(
        """
        ```mermaid
        flowchart LR
            A[Legal PDFs / TXT] --> B[Document Loader & Cleaner]
            B --> C[Recursive Text Splitter]
            C --> D[SentenceTransformer Embeddings]
            D --> E[(ChromaDB Vector Store)]
            F[User Legal Question] --> G[Query Embedder]
            G --> H[Semantic Cosine Search]
            E --> H
            H --> I[Evidence & Metadata Filtering]
            I --> J[Grounding Prompt & Ollama LLM]
            J --> K[Grounded Answer + Citations]
        ```
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Key Technical Highlights")
    st.markdown(
        """
        - **Modular Python Architecture:** Fully decoupled loading, processing, embedding, retrieval, and generation layers.
        - **Zero-Cost Embeddings:** Runs `sentence-transformers/all-MiniLM-L6-v2` locally on CPU (384-dimensional dense vectors).
        - **Persistent Storage:** Uses embedded ChromaDB with SHA-256 chunk deduplication.
        - **Strict Legal Grounding:** Custom system prompt prevents hallucinations and enforces "Insufficient Evidence" disclosures.
        - **Traceable Source Citations:** Maps every answer back to specific document titles, courts, pages, and verbatim excerpts.
        - **Offline Fallback Engine:** Operates seamlessly as an evidence retrieval engine even when local LLM server is offline.
        """
    )
