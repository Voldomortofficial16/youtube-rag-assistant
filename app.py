import streamlit as st

from groq import Groq

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ---------------------------------
# PAGE CONFIG
# ---------------------------------

st.set_page_config(
    page_title="YouTube NLP Learning Assistant",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 YouTube NLP Learning Assistant")

st.markdown("""
Ask questions from the CampusX NLP Playlist using
Retrieval-Augmented Generation (RAG).

**Tech Stack:** LangChain • FAISS • Sentence Transformers • Groq • Streamlit
""")


# ---------------------------------
# GROQ CLIENT
# ---------------------------------

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)


# ---------------------------------
# LOAD FAISS
# ---------------------------------

@st.cache_resource
def load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        ".",   # current directory
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db


vectorstore = load_vectorstore()


# ---------------------------------
# RAG FUNCTION
# ---------------------------------

def ask_rag(question):

    docs = vectorstore.similarity_search(
        question,
        k=5
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are an expert NLP tutor.

Answer ONLY from the provided context.

If the answer is not available in the context,
reply with:

"I could not find this information in the playlist."

Context:
{context}

Question:
{question}

Provide:
1. Explanation
2. Example
3. Key Takeaway
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# ---------------------------------
# UI
# ---------------------------------

question = st.text_input(
    "Ask a Question",
    placeholder="Example: What is Tokenization?"
)

if st.button("Get Answer"):

    if question.strip() == "":
        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching playlist and generating answer..."):

            answer = ask_rag(question)

        st.subheader("Answer")

        st.write(answer)


# ---------------------------------
# SIDEBAR
# ---------------------------------

with st.sidebar:

    st.header("Project Info")

    st.markdown("""
This assistant is built using:

- LangChain
- FAISS Vector Database
- Sentence Transformers
- Groq Llama 3
- Streamlit

Knowledge Source:

CampusX NLP Playlist
""")
