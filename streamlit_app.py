import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-flash-lite-latest")

st.set_page_config(
page_title="Resume RAG Analyzer",
page_icon="📄",
layout="wide"
)
st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b
    );
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

h1 {
    color: #38bdf8;
}

.stButton > button {
    border-radius: 12px;
}

.stChatMessage {
    border-radius: 15px;
    padding: 12px;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<h1 style='color:#60A5FA;'>
🤖 Resume RAG Analyzer
</h1>
""", unsafe_allow_html=True)
st.caption(
    "AI-powered Resume Analysis using LangChain, FAISS and Gemini"
)
st.markdown("""
<div style="
padding:20px;
border-radius:15px;
background:rgba(255,255,255,0.05);
backdrop-filter: blur(10px);
">
</div>
""", unsafe_allow_html=True)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:

    st.title("🤖 Resume AI")

    st.success("✅ Ready")

    st.info("📄 Upload Resume")

    st.info("⚡ gemini-flash-lite-latest")

def extract_text(pdf_file):

    text = ""

    reader = PdfReader(pdf_file)

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text

    return text
def split_text(text):

    splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
    )

    return splitter.split_text(text)
@st.cache_resource
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    return vector_store
def answer_question(query, vector_store):

    docs = vector_store.similarity_search(
    query,
    k=3
    )

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
    You are a Resume Analysis Assistant.

    Resume Context:
    {context}

    Question:
    {query}

    Answer ONLY using the resume context.
    """

    response = model.generate_content(prompt)

    return response.text
st.markdown("""
<div style="
padding:20px;
border-radius:15px;
background:rgba(255,255,255,0.05);
backdrop-filter: blur(10px);
margin-bottom:15px;
">
<h3>📄 Upload Your Resume</h3>
<p>Get AI-powered insights from your resume.</p>
</div>
""", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)
if uploaded_file:

    text = extract_text(uploaded_file)

    st.success("Resume uploaded successfully!")

    with st.expander("View Resume"):
        st.write(text[:3000])

    chunks = split_text(text)
    with st.sidebar:

     st.title("📄 Resume RAG")

    st.markdown("---")

    st.success("✅ Resume Uploaded")

    st.info(f"Chunks Created: {len(chunks)}")

    st.info("Model: Gemini 2.5 Flash")

    st.markdown("---")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.write(f"Total Chunks: {len(chunks)}")

    if "vector_store" not in st.session_state:
      st.session_state.vector_store = create_vector_store(chunks)

    vector_store = st.session_state.vector_store

    st.success("✅ FAISS Vector Store Created")
    col1, col2, col3 = st.columns(3)

    with col1:
     st.metric("Chunks", len(chunks))

    with col2:
     st.metric("Words", len(text.split()))

    with col3:
     st.metric("Pages", len(PdfReader(uploaded_file).pages))
    
    col1, col2, col3 = st.columns(3)

    with col1:
     st.metric("Chunks", len(chunks))

    with col2:
     st.metric("Status", "Ready")

    with col3:
     st.metric("Model", "Gemini 2.5")

    st.markdown("""
    ### Sample Questions

    - What are the candidate's skills?
    - What projects has the candidate completed?
    - What is the educational background?
    - What technologies does the candidate know?
    """)

if st.button("Generate Resume Summary"):

    summary = answer_question(
        """
        Provide a detailed professional summary of this candidate.

        Include:
        - Education
        - Technical Skills
        - Projects
        - Experience
        - Tools and Technologies

        Write the summary in 5-8 bullet points.
        """,
        vector_store
    )

    st.subheader("📋 Resume Summary")
    st.info(summary)


if st.button("Extract Skills"):

    skills = answer_question(
        """
        Extract all technical skills from the resume.
        Group them by:
        - Programming Languages
        - Databases
        - Tools
        - Frameworks
        - Cloud Technologies
        """,
        vector_store
    )

    st.subheader("🛠 Skills")
    st.info(skills)


# ATS SECTION STARTS HERE

st.subheader("🎯 ATS Resume Match")

job_description = st.text_area(
    "Paste Job Description Here"
)

if uploaded_file and job_description and st.button("ATS Match Score"):

    result = answer_question(
        f"""
        Compare the resume with this job description:

        {job_description}

        Give:
        1. ATS Score out of 100
        2. Missing Skills
        3. Strengths
        4. Recommendations
        """,
        vector_store
    )

    st.subheader("📊 ATS Analysis")
    st.success(result)
query = st.chat_input(
    "Ask a question about the resume..."
)

if query:

    with st.spinner("Analyzing Resume..."):

        answer = answer_question(
            query,
            vector_store
        )

    st.session_state.chat_history.append(
        ("You", query)
    )

    st.session_state.chat_history.append(
        ("AI", answer)
    )

for role, message in st.session_state.chat_history:

    if role == "You":

        with st.chat_message("user"):
            st.write(message)

    else:

        with st.chat_message("assistant"):
            st.write(message)
st.caption(
    "© 2026 Atharva Borkar | Resume RAG Analyzer | Powered by Gemini 2.5, LangChain & FAISS"
)