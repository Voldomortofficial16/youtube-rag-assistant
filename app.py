import streamlit as st

from groq import Groq

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


st.set_page_config(
    page_title="YouTube NLP Assistant",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 YouTube NLP Learning Assistant")

st.write(
    "Ask questions from the CampusX NLP Playlist"
)

# -----------------------------
# GROQ
# -----------------------------

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)

# -----------------------------
# EMBEDDINGS
# -----------------------------

@st.cache_resource
def load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db

vectorstore = load_vectorstore()

# -----------------------------
# RAG FUNCTION
# -----------------------------

def ask_rag(question):

    docs = vectorstore.similarity_search(
        question,
        k=5
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are an NLP tutor.

Answer only using the provided context.

If the answer is not found,
say:

I could not find this information in the playlist.

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ]
    )

    return response.choices[0].message.content


question = st.text_input(
    "Ask a Question"
)

if st.button("Get Answer"):

    with st.spinner("Thinking..."):

        answer = ask_rag(question)

    st.subheader("Answer")

    st.write(answer)
