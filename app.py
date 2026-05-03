import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import glob
import numpy as np

st.set_page_config(page_title="Travel Destinations RAG", page_icon="✈️", layout="wide")

@st.cache_resource(show_spinner="Loading documents...")
def load_index():
    files = glob.glob("documents/*.txt")
    documents = []
    sources = []
    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        filename = os.path.basename(filepath).replace(".txt", "").replace("_", " ").title()
        chunks = [text[i:i+150] for i in range(0, len(text), 125)]
        for chunk in chunks:
            if len(chunk.strip()) > 20:
                documents.append(chunk.strip())
                sources.append(filename)
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(documents)
    return vectorizer, matrix, documents, sources

vectorizer, matrix, documents, sources = load_index()

st.sidebar.title("Travel RAG")
page = st.sidebar.radio("Navigation", ["Home", "Search"])

if page == "Home":
    st.title("✈️ Travel Destinations Explorer")
    st.markdown("Welcome to the AI-powered travel search engine!")
    st.markdown("**How it works:**")
    st.markdown("1. Travel blog articles from worldtraveler.com are split into small chunks")
    st.markdown("2. Each chunk is indexed using TF-IDF vectorization")
    st.markdown("3. When you search, your query is matched against the most relevant chunks")
    st.markdown("4. The most relevant travel information is returned to you")
    st.markdown("---")
    st.markdown("### 🌍 Destinations in this database:")
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
    st.title("🔍 Search Travel Destinations")
    query = st.text_input("What do you want to know?", placeholder="e.g. Where can I go diving?")
    num_results = st.slider("Number of results", min_value=1, max_value=6, value=3)
    if query:
        with st.spinner("Searching..."):
            query_vec = vectorizer.transform([query])
            scores = cosine_similarity(query_vec, matrix).flatten()
            top_indices = np.argsort(scores)[::-1][:num_results]
        st.markdown(f"### Top {num_results} results for: *{query}*")
        for i, idx in enumerate(top_indices):
            with st.expander(f"Result {i+1} — {sources[idx]}", expanded=True):
                st.write(documents[idx])
    else:
        st.markdown("### 💡 Example searches:")
        st.markdown("- Where can I go scuba diving?")
        st.markdown("- Which destination is best for budget travelers?")
        st.markdown("- Where can I see ancient ruins?")
        st.markdown("- What is the best time to visit Iceland?")
        st.markdown("- Which places are safe for solo travelers?")
        st.markdown("- Which destination is best for budget travelers?")
        st.markdown("- Where can I see ancient ruins?")
        st.markdown("- What is the best time to visit Iceland?")
        st.markdown("- Which places are safe for solo travelers?")
