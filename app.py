import os
import uuid
import chromadb
import streamlit as st
from dotenv import load_dotenv
from google import genai

from agent import run_research_agent
from utils import (
    get_uploaded_file_signature,
    index_chunks,
    process_pdfs,
)


# =========================================================
# Configuration
# =========================================================

load_dotenv()

st.set_page_config(
    page_title="Research Assistant",
    page_icon="📚",
    layout="wide",
)


# =========================================================
# Session state
# =========================================================

DEFAULT_STATE = {
    "pdfs": [],
    "chunks": [],
    "uploaded_signature": None,
    "papers_indexed": False,
    "response": None,
    "tool_log": [],
    "last_question": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    
# =========================================================
# Clients
# =========================================================

@st.cache_resource
def get_gemini_client():
    return genai.Client(
        api_key=os.getenv("GEMINI_API_KEY")
    )


@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(
        path="./chroma_db"
    )


gemini_client = get_gemini_client()
chroma_client = get_chroma_client()

collection = chroma_client.get_or_create_collection(
    name=("research" + st.session_state.session_id.replace("-", ""))
)

# =========================================================
# Header
# =========================================================

st.title("Research Assistant")

st.caption(
    "Ask questions, compare papers, and trace the evidence "
    "used to construct each answer."
)

st.divider()


# =========================================================
# Main layout
# =========================================================

left_column, right_column = st.columns(
    [0.42, 0.58],
    gap="large",
)


# =========================================================
# LEFT COLUMN
# Upload + indexing + question + agent activity
# =========================================================

with left_column:

    st.subheader("Research workspace")

    # -----------------------------------------------------
    # Upload papers
    # -----------------------------------------------------

    uploaded_files = st.file_uploader(
        "Upload research papers",
        type="pdf",
        accept_multiple_files=True,
    )

    if uploaded_files:

        current_signature = get_uploaded_file_signature(
            uploaded_files
        )

        if (
            current_signature
            != st.session_state.uploaded_signature
        ):

            st.session_state.pdfs = [
                {
                    "file_name": file.name,
                    "data": file.getvalue(),
                }
                for file in uploaded_files
            ]

            with st.spinner("Reading papers..."):
                st.session_state.chunks = process_pdfs(
                    st.session_state.pdfs
                )

            st.session_state.uploaded_signature = (
                current_signature
            )

            st.session_state.papers_indexed = False
            st.session_state.response = None
            st.session_state.tool_log = []
            st.session_state.last_question = None


    # -----------------------------------------------------
    # Uploaded paper list
    # -----------------------------------------------------

    if st.session_state.pdfs:

        st.caption("UPLOADED PAPERS")

        for pdf in st.session_state.pdfs:
            st.markdown(
                f"📄 **{pdf['file_name']}**"
            )

        st.caption(
            f"{len(st.session_state.pdfs)} paper(s) · "
            f"{len(st.session_state.chunks)} chunks"
        )


        # -------------------------------------------------
        # Indexing
        # -------------------------------------------------

        if st.session_state.papers_indexed:

            st.success(
                "Ready to search",
                icon="✅",
            )

        else:

            if st.button(
                "Index papers",
                type="primary",
                use_container_width=True,
            ):

                with st.spinner(
                    "Creating embeddings and indexing..."
                ):

                    index_chunks(
                        chunks=st.session_state.chunks,
                        gemini_client=gemini_client,
                        collection=collection,
                    )

                st.session_state.papers_indexed = True

                st.rerun()

    else:

        st.info(
            "Upload one or more PDFs to begin."
        )


    # -----------------------------------------------------
    # Question input
    # -----------------------------------------------------

    if st.session_state.papers_indexed:

        st.divider()

        with st.form("research_form"):

            question = st.text_area(
                "Ask a question",
                placeholder=(
                    "Compare the approaches used in these papers..."
                ),
                height=130,
            )

            submitted = st.form_submit_button(
                "Research",
                type="primary",
                use_container_width=True,
            )


        if submitted and question.strip():

            st.session_state.last_question = question

            with st.spinner(
                "Researching..."
            ):

                response, tool_log = run_research_agent(
                    question=question,
                    gemini_client=gemini_client,
                    collection=collection,
                    pdfs=st.session_state.pdfs,
                )

            st.session_state.response = response
            st.session_state.tool_log = tool_log


    # -----------------------------------------------------
    # Agent activity
    # -----------------------------------------------------

    if st.session_state.tool_log:

        st.divider()

        st.caption("AGENT ACTIVITY")

        tool_labels = {
            "search_one_paper":
                "Search one paper",

            "search_all_papers":
                "Search across papers",

            "get_page_context":
                "Read more context",
        }

        for index, tool in enumerate(
            st.session_state.tool_log,
            start=1,
        ):

            tool_name = tool["name"]

            label = tool_labels.get(
                tool_name,
                tool_name,
            )

            with st.expander(
                f"{index}. 🔎 {label}",
                expanded=False,
            ):

                if tool.get("args"):
                    st.json(
                        tool["args"]
                    )


# =========================================================
# RIGHT COLUMN
# Answer only
# =========================================================

with right_column:

    st.subheader("Answer")

    response = st.session_state.response

    if response and response.text:

        if st.session_state.last_question:

            st.caption(
                st.session_state.last_question
            )

        st.markdown(
            response.text
        )

    else:

        st.markdown(
            """
            <div style="
                padding: 2rem;
                border: 1px solid rgba(128, 128, 128, 0.18);
                border-radius: 14px;
                min-height: 420px;
            ">
                <p style="
                    opacity: 0.5;
                    margin: 0;
                    font-size: 1rem;
                ">
                    Your research answer will appear here.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )