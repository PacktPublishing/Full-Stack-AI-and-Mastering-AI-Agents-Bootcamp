import streamlit as st
import faiss
import numpy as np
import PyPDF2
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

# Load AI Model
llm = OllamaLLM(model="mistral")  # Change to "llama3" or another Ollama model

# Load Hugging Face Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize FAISS Vector Database
index = faiss.IndexFlatL2(384)  # Vector dimension for MiniLM
vector_store = {}
summary_text = ""

# Function to extract text from PDFs
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to store text in FAISS
def store_in_faiss(text, filename):
    global index, vector_store
    st.write(f"📥 Storing document '{filename}' in FAISS...")

    # Split text into chunks
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = splitter.split_text(text)

    # Convert text into embeddings
    vectors = embeddings.embed_documents(texts)
    vectors = np.array(vectors, dtype=np.float32)

    # Store in FAISS
    index.add(vectors)
    vector_store[len(vector_store)] = (filename, texts)

    return "✅ Document stored successfully!"

# Function to generate AI summary
def generate_summary(text):
    global summary_text
    st.write("📝 Generating AI Summary...")
    summary_text = llm.invoke(f"Summarize the following document:\n\n{text[:3000]}")  # Limiting input size
    return summary_text


# Function to retrieve relevant chunks and answer questions
def retrieve_and_answer(query):
    global index, vector_store

    # Convert query into embedding
    query_vector = np.array(embeddings.embed_query(query), dtype=np.float32).reshape(1, -1)

    # Search FAISS
    D, I = index.search(query_vector, k=2)  # Retrieve top 2 similar chunks

    context = ""
    for idx in I[0]:
        if idx in vector_store:
            context += " ".join(vector_store[idx][1]) + "\n\n"

    if not context:
        return "🤖 No relevant data found in stored documents."

    # Ask AI to generate an answer
    return llm.invoke(f"Based on the following document context, answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:")

# Function to allow file download
def download_summary():
    if summary_text:
        st.download_button(
            label="📥 Download Summary",
            data=summary_text,
            file_name="AI_Summary.txt",
            mime="text/plain"
        )

# Streamlit Web UI
st.title("📄 AI Document Reader & Q&A Bot")
# st.write("Upload a PDF and ask questions based on its content!")
st.write("Upload a PDF and get an AI-generated summary & Q&A!")

# File uploader for PDF
uploaded_file = st.file_uploader("📂 Upload a PDF Document", type=["pdf"])
if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    store_message = store_in_faiss(text, uploaded_file.name)
    st.write(store_message)

    # Generate AI Summary
    summary = generate_summary(text)
    st.subheader("AI-Generated Summary")
    st.write(summary)

    # Enable File Download for Summary
    download_summary()

# User input for Q&A
query = st.text_input("❓ Ask a question based on the uploaded document:")
if query:
    answer = retrieve_and_answer(query)
    st.subheader("🤖 AI Answer:")
    st.write(answer)










