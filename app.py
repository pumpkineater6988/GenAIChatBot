# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 14:57:33 2025

@author: sahil
"""

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
st.title("📊 GenAI Chatbot")

# ---- Session State ----
for key in ["retriever", "chat_history", "dataframe"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "chat_history" else None

# ---- Numeric Query Handler ----
def handle_numeric_query(query, df):
    if df is None:
        return None
    query = query.lower()

    # SUM or TOTAL
    if "sum" in query or "total" in query:
        for col in df.select_dtypes(include='number').columns:
            if col.lower() in query:
                total = df[col].sum()
                return f"🔢 Total sum of `{col}` is: {total}"

    # COUNT
    if "count" in query or "how many" in query:
        for col in df.columns:
            if col.lower() in query:
                count = df[col].nunique()
                return f"🔢 Unique count of `{col}` is: {count}"

    return None

# ---- File Upload ----
uploaded_file = st.file_uploader("Upload PDF, CSV, or Excel", type=["pdf", "csv", "xlsx"])
if uploaded_file and st.button("📥 Upload & Process File"):
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
        st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        st.success("✅ File processed and embedded!")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

# ---- Question Answering ----
if st.session_state.retriever:
    user_query = st.text_input("Ask your question")

    if st.button("Ask") and user_query:
        try:
            # First try to handle numeric questions with pandas
            response = handle_numeric_query(user_query, st.session_state.dataframe)

            if response:
                st.write("🤖 Bot:", response)
                st.session_state.chat_history.append((user_query, response))
            else:
                docs = st.session_state.retriever.get_relevant_documents(user_query)
                context = "\n".join([doc.page_content for doc in docs])

                messages: list[ChatCompletionMessageParam] = [
                    {"role": "system", "content": "You are a helpful financial assistant. If answering with tables, return them in JSON format."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
                ]

                result = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages
                )
                answer = getattr(result.choices[0].message, "content", None)
                st.session_state.chat_history.append((user_query, answer))

                st.markdown("### 🤖 GPT-4o Answer")
                if answer and "```json" in answer:
                    try:
                        match = re.search(r"```json(.*?)```", answer, re.DOTALL)
                        if match:
                            table_json = match.group(1).strip()
                            table_data = json.loads(table_json)
                            st.dataframe(pd.DataFrame(table_data))
                        else:
                            st.write(answer)
                    except:
                        st.write(answer)
                else:
                    st.write(answer)

        except Exception as e:
            st.error(f"❌ Error from OpenAI: {e}")

# ---- Chat History ----
if st.session_state.chat_history:
    with st.expander("🕘 Chat History"):
        for i, (q, a) in enumerate(st.session_state.chat_history):
            st.markdown(f"**Q{i+1}: {q}**")
            st.markdown(f"**A{i+1}:** {a}")

    if st.button("📤 Export Chat to CSV"):
        try:
            csv_data = pd.DataFrame(st.session_state.chat_history, columns=["Question", "Answer"])
            csv_buffer = StringIO()
            csv_data.to_csv(csv_buffer, index=False)
            st.download_button("Download Chat CSV", csv_buffer.getvalue(), file_name="chat_history.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error exporting: {e}")