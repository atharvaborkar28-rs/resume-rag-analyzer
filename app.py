import google.generativeai as genai
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

# Step 1 - Extract text from PDF
def extract_text(pdf_file):
    text = ""

    reader = PdfReader(pdf_file)

    for page in reader.pages:
        text += page.extract_text()

    return text


# Step 2 - Split text into chunks
def split_text(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)
    

    return chunks

# Step 3 - Create Embeddings
def create_embeddings():

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embedding_model


# Step 4 - Create FAISS Vector Store
from langchain_community.vectorstores import FAISS

def create_vector_store(chunks):

    embeddings = create_embeddings()

    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    return vector_store

# TESTING RAG + GEMINI


def retrieve_docs(query, vector_store):

    docs = vector_store.similarity_search(
        query,
        k=5
    )



    for i, doc in enumerate(docs):
        print(f"\nDOC {i+1}")
        print(doc.page_content)
        print("-" * 50)

    return docs

def answer_question(query, docs):

    context = "\n".join([doc.page_content for doc in docs])

    
    print(context)

    prompt = f"""
    You are a Resume Analysis Assistant.

    Resume Context:
    {context}

    Question:
    {query}

    Answer ONLY using the resume context provided.
    If partial information exists, answer with whatever is available.
    """

    response = model.generate_content(prompt)

    return response.text

# MAIN


text = extract_text("resume.pdfe")

chunks = split_text(text)

print("=" * 50)

print(f"\nNumber of Chunks: {len(chunks)}")

vector_store = create_vector_store(chunks)

print("\n🤖 Resume Chatbot Ready!")
print("Type 'exit' to quit.\n")

while True:

    query = input("You: ")

    if query.lower() == "exit":
        print("\nBot: Goodbye! 👋")
        break

    docs = retrieve_docs(query, vector_store)

    answer = answer_question(query, docs)

    print("\nBot:")
    print(answer)
    print("\n" + "=" * 60 + "\n")