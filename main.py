import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# --- LOCAL TESTING MODE ---
# Put your real API key right here for now. 
os.environ["GOOGLE_API_KEY"] = "AIzaSyBqsy6DImFlWDjXxZKL8bZ6foi8hji_BdE" # <-- YOUR REAL KEY HERE

# (We will delete the line above and uncomment the line below ONLY when you upload to GitHub)
# os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

st.set_page_config(page_title="Control Tower AI", page_icon="🗼", layout="wide")


# --- 2. CACHE THE SEMANTIC ENGINE ---
@st.cache_resource(show_spinner="Booting Local Semantic Intelligence...")
def load_semantic_engine():
    # 1. Load Local Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # 2. Load FAISS Database
    vector_db = FAISS.load_local(
        "faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # 3. Initialize Cloud LLM for Reasoning
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
    
    # 4. High-Precision Retriever
    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={'k': 4})
    
    # 5. The "Sniper" Prompt
    system_prompt = (
        "You are the strict, highly precise Lead Solutions Architect for ClickPost. "
        "Answer the user's question using ONLY the provided technical documentation.\n\n"
        "RULES:\n"
        "1. STRICT PRECISION: Answer EXACTLY what is asked.\n"
        "2. FORMATTING: Use bold headings and clear bullet points. Include exact status codes.\n"
        "3. NO HALLUCINATION: If the exact answer is missing from the context, output: 'I do not have the exact answer in the documentation.' Do not guess.\n\n"
        "CONTEXT:\n{context}"
    )
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    document_chain = create_stuff_documents_chain(llm, prompt_template)
    return create_retrieval_chain(retriever, document_chain)

# Initialize the chain
semantic_chain = load_semantic_engine()

# --- 3. UI DASHBOARD ---
with st.sidebar:
    st.header("🗼 Engine Specs")
    st.success("Search: Local Semantic (mpnet-v2)")
    st.success("Reasoning: Gemini 2.5 Flash")
    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = [{"role": "assistant", "content": "System reset. What Control Tower logic do you need?"}]
        st.rerun()

st.title("Control Tower Enterprise AI 🤖")
st.markdown("Precision data extraction from internal operational documentation.")

# --- 4. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Elite System Online. Ask about a widget, SLA formula, or metric."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("E.g., What are the key conditions for the Forward RTO Without Attempt widget?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Retrieving semantic blocks..."):
            try:
                response = semantic_chain.invoke({"input": prompt})
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"🚨 Engine Error: {str(e)}")