import fitz  # PyMuPDF for extracting text from the PDF
from sentence_transformers import SentenceTransformer
import faiss
import openai
import os
import warnings
import streamlit as st
import pickle
import hashlib
import os
from dotenv import load_dotenv
load_dotenv() 

# Helper function to generate a hash for caching purposes
def get_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# Step 1: Extract text from the PDF
def extract_text_from_pdf(pdf_path):
    text_data = []
    try:
        with fitz.open(pdf_path) as pdf:
            for page_number in range(pdf.page_count):
                page = pdf.load_page(page_number)
                text = page.get_text()
                text_data.append(text)
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
    return text_data

# Step 2: Split the text into manageable chunks for embedding
def split_text_into_chunks(text_data, chunk_size=500):
    chunks = []
    for page_text in text_data:
        words = page_text.split()
        for i in range(0, len(words), chunk_size):
            chunks.append(' '.join(words[i:i + chunk_size]))
    return chunks

# Step 3: Embed text chunks using Sentence Transformers
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text_chunks(chunks):
    return model.encode(chunks, convert_to_tensor=False)

# Step 4: Store embeddings in FAISS for similarity search
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Step 5: Query Processing
def find_similar_chunks(query, index, chunks, top_k=45):
    query_embedding = model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

# Step 6: Use GPT-4 (or ChatGPT) to generate a response with streaming

def generate_response(retrieved_chunks, user_query):
    context = "".join(retrieved_chunks)
    prompt = f"Carl Jung assistant. You answer questions using citations from the information provided in the context. In the context provided to you, there will be the page number and name of the work, include both in your citaion. You are a Jungain expeort. based on the following context:{context}Answer the user's question: {user_query}"
    client = openai.OpenAI()
    client.api_key = os.getenv("OPENAI_API_KEY")
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            response += chunk.choices[0].delta.content
    return response.strip()

# Main flow with Streamlit
def password_authentication():
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    if not st.session_state['authenticated']:
        password = st.text_input("Enter the password to access the chatbot:", type="password")
        if password == os.getenv("PASSWORD"):
            st.session_state['authenticated'] = True
            st.success("Password correct! You can now use the chatbot.")
            return True  # Rerun the app to show the chatbot interface
        elif password:
            st.error("Incorrect password. Please try again.")
        return False
    return True

def main():
    if not password_authentication():
        return

    st.title("Carl Jung Chatbot - RAG Model")
    st.markdown("This chatbot allows you to ask questions about Carl Jung's works based on his collected writings.")

    pdf_path = "Data/C.G-Jung-Collected-Works-in-one-file.pdf"
    cache_file = "cache_data.pkl"
    pdf_hash = get_hash(pdf_path)

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        if cache.get('pdf_hash') == pdf_hash:
            text_data = cache['text_data']
            chunks = cache['chunks']
            embeddings = cache['embeddings']
            index = cache['index']
            st.success("Loaded cached data successfully!")
        else:
            st.warning("PDF has changed, recalculating data...")
            with st.spinner("Extracting text from the PDF..."):
                text_data = extract_text_from_pdf(pdf_path)
            with st.spinner("Splitting text into chunks..."):
                chunks = split_text_into_chunks(text_data)
            with st.spinner("Embedding text chunks..."):
                embeddings = embed_text_chunks(chunks)
            with st.spinner("Creating FAISS index..."):
                index = create_faiss_index(embeddings)
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'pdf_hash': pdf_hash,
                    'text_data': text_data,
                    'chunks': chunks,
                    'embeddings': embeddings,
                    'index': index
                }, f)
    else:
        with st.spinner("Extracting text from the PDF..."):
            text_data = extract_text_from_pdf(pdf_path)
        with st.spinner("Splitting text into chunks..."):
            chunks = split_text_into_chunks(text_data)
        with st.spinner("Embedding text chunks..."):
            embeddings = embed_text_chunks(chunks)
        with st.spinner("Creating FAISS index..."):
            index = create_faiss_index(embeddings)
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'pdf_hash': pdf_hash,
                'text_data': text_data,
                'chunks': chunks,
                'embeddings': embeddings,
                'index': index
            }, f)

    st.success("Ready to answer questions!")
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Input box at the top of the page
    user_query = st.text_input("Ask a question about Carl Jung's works:", key="user_input")
    if user_query:
        retrieved_chunks = find_similar_chunks(user_query, index, chunks)
        response = generate_response(retrieved_chunks, user_query)
        st.session_state['chat_history'].append(("JungBot", response))
        st.session_state['chat_history'].append(("You", user_query))
                

    # Display chat history in reverse order (newest at the top)
    if st.session_state['chat_history']:
        for speaker, message in reversed(st.session_state['chat_history']):
            st.markdown(f"**{speaker}:** {message}")

if __name__ == "__main__":
    # Suppress warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()