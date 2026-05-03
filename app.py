import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain.embeddings.base import Embeddings
import os
import glob

st.set_page_config(page_title="Travel Destinations RAG", page_icon="✈️", layout="wide")

class ChromaDefaultEmbeddings(Embeddings):
    def __init__(self):
        self.ef = DefaultEmbeddingFunction()
    def embed_documents(self, texts):
        return self.ef(texts)
    def embed_query(self, text):
        return self.ef([text])[0]

@st.cache_resource(show_spinner="Loading documents...")
def load_vectorstore():
    files = glob.glob("documents/*.txt")
    documents = []
    metadatas = []
    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        filename = os.path.basename(filepath).replace(".txt", "").replace("_", " ").title()
        documents.append(text)
        metadatas.append({"source": filename})
    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=25)
    chunks = []
    chunk_metas = []
    for doc, meta in zip(documents, metadatas):
        splits = splitter.split_text(doc)
        chunks.extend(splits)
        chunk_metas.extend([meta] * len(splits))
    embeddings = ChromaDefaultEmbeddings()
    vectorstore = Chroma.from_texts(chunks, embeddings, metadatas=chunk_metas)
    return vectorstore

st.sidebar.title("Travel RAG")
page = st.sidebar.radio("Navigation", ["Home", "Search"])

vectorstore = load_vectorstore()

if page == "Home":
    st.title("Travel Destinations Explorer")
    st.markdown("Welcome to the AI-powered travel search engine!")
    st.markdown("This app uses RAG to find information from travel blog articles.")
    st.markdown("**How it works:**")
    st.markdown("1. Travel blog articles are split into small chunks")
    st.markdown("2. Each chunk is converted into a semantic embedding vector")
    st.markdown("3. When you search, your query is matched against the most relevant chunks")
    st.markdown("4. The most relevant travel information is returned to you")
    st.markdown("---")
    st.markdown("### Destinations in this database:")
    cols = st.columns(3)
    destinations = [
        "Buton, Indonesia", "Egypt", "Jordan",
        "Oman", "New Zealand", "Palau",
        "India", "Spain", "Zhangjiajie, China", "Iceland"
    ]
    for i, name in enumerate(destinations):
        with cols[i % 3]:
            st.markdown(f"**{name}**")
    st.markdown("---")
    st.info("Use the sidebar to go to the Search page!")

elif page == "Search":
    st.title("Search Travel Destinations")
    query = st.text_input("What do you want to know?", placeholder="e.g. Where can I go diving?")
    num_results = st.slider("Number of results", min_value=1, max_value=6, value=3)
    if query:
        with st.spinner("Searching..."):
            results = vectorstore.similarity_search(query, k=num_results)
        st.markdown(f"### Top {len(results)} results for: {query}")
        for i, doc in enumerate(results):
            source = doc.metadata.get("source", "Unknown")
            with st.expander(f"Result {i+1} — {source}", expanded=True):
                st.write(doc.page_content)
    else:
        st.markdown("### Example searches:")
        st.markdown("- Where can I go scuba diving?")
        st.markdown("- Which destination is best for budget travelers?")
        st.markdown("- Where can I see ancient ruins?")
        st.markdown("- What is the best time to visit Iceland?")
        st.markdown("- Which places are safe for solo travelers?")
        st.markdown("- Which destination is best for budget travelers?")
        st.markdown("- Where can I see ancient ruins?")
        st.markdown("- What is the best time to visit Iceland?")
        st.markdown("- Which places are safe for solo travelers?")
