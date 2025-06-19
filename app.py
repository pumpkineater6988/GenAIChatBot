import streamlit as st
import tempfile
import chardet
import pandas as pd
import os
import json
import re
from openai import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from io import StringIO
from dotenv import load_dotenv
from openai.types.chat import ChatCompletionMessageParam

# ---- Load Environment Variables ----
load_dotenv()

# ---- OpenAI Client ----
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---- Streamlit UI ----
st.set_page_config(page_title="GenAI Chatbot", layout="centered")
st.markdown("""
    <style>
    .chat-message {
        padding: 8px 16px;
        border-radius: 16px;
        margin-bottom: 8px;
        max-width: 80%;
        word-break: break-word;
        font-size: 1.1em;
        color: #fff;
    }
    .user-message {
        background-color: #1a237e; /* dark blue */
        align-self: flex-end;
        margin-left: auto;
    }
    .bot-message {
        background-color: #263238; /* dark gray */
        align-self: flex-start;
        margin-right: auto;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 4px;
        min-height: 400px;
        margin-bottom: 16px;
    }
    </style>
""", unsafe_allow_html=True)
st.title("💬 GenAI Chatbot")

# ---- Session State ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # (role, message)
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "dataframe" not in st.session_state:
    st.session_state.dataframe = None

# ---- File Upload (Optional) ----
st.sidebar.header("📁 Optional: Upload Data File")
uploaded_file = st.sidebar.file_uploader("Upload PDF, CSV, or Excel", type=["pdf", "csv", "xlsx"])
if uploaded_file and st.sidebar.button("Process File"):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        docs = []
        df = None

        if uploaded_file.name.endswith(".csv"):
            with open(tmp_path, 'rb') as f:
                encoding = chardet.detect(f.read())['encoding']
            df = pd.read_csv(tmp_path, encoding=encoding)
            docs = [Document(page_content=str(row.to_json())) for _, row in df.iterrows()]
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(tmp_path)
            docs = [Document(page_content=str(row.to_json())) for _, row in df.iterrows()]
        elif uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
        st.session_state.dataframe = df
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        st.sidebar.success("✅ File processed and embedded!")
    except Exception as e:
        st.sidebar.error(f"❌ Error processing file: {e}")

# ---- Chat Interface ----
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for role, message in st.session_state.chat_history:
    if role == "user":
        st.markdown(f'<div class="chat-message user-message">🧑‍💻 {message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message bot-message">🤖 {message}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---- User Input ----
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your message...", key="user_input")
    submitted = st.form_submit_button("Send")

# Only process the message if the form was just submitted and input is not empty
if submitted and user_input:
    try:
        context = ""
        if st.session_state.retriever:
            docs = st.session_state.retriever.get_relevant_documents(user_input)
            context = "\n".join([doc.page_content for doc in docs])
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are a helpful assistant. If context is provided, use it to answer. If not, answer as best as you can. Be conversational and concise."}
        ]
        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}"})
        # Only use last 3 exchanges for context
        for role, msg in st.session_state.chat_history[-3:]:
            if role == "user":
                messages.append({"role": "user", "content": msg})
            else:
                messages.append({"role": "assistant", "content": msg})
        messages.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            result = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
        answer = getattr(result.choices[0].message, "content", None)
        if not answer:
            answer = "Sorry, I couldn't generate a response."
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("assistant", answer))
    except Exception as e:
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("assistant", f"❌ Error: {e}"))

# ---- Chat History Export ----
with st.sidebar.expander("🕘 Chat History & Export"):
    if st.session_state.chat_history:
        if st.button("📤 Export Chat to CSV"):
            try:
                csv_data = pd.DataFrame(
                    [(role, msg) for role, msg in st.session_state.chat_history],
                    columns=["Role", "Message"]
                )
                csv_buffer = StringIO()
                csv_data.to_csv(csv_buffer, index=False)
                st.download_button("Download Chat CSV", csv_buffer.getvalue(), file_name="chat_history.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Error exporting: {e}")
