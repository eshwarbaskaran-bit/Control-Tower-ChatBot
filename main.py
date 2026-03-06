import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- LOCAL TESTING MODE ---
# Put your NEW API key right here (since the old one was exposed!)
os.environ["GOOGLE_API_KEY"] = "AIzaSyAk1qLFmUVTD_6Pjlxl_uD5d42TieTzqJw"

st.set_page_config(page_title="Control Tower AI", page_icon="🗼", layout="wide")

# --- 2. CACHE THE SEMANTIC ENGINE ---
@st.cache_resource(show_spinner="Booting Local Semantic Intelligence...")
def load_semantic_engine():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    
    # K increased to 8, switched to similarity so it reads all relevant logic
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={'k': 8})
    
    # --- MEMORY REWRITER ---
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history, "
        "formulate a standalone question which can be understood without the chat history. "
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # --- THE 3-BULLET PRODUCT OPS PROMPT ---
    system_prompt = (
        "# Role\n"
        "- You are a Senior Product Operations Analyst working at ClickPost.\n"
        "- Your objective is to explain Control Tower and logistics logic with extreme brevity.\n"
        "- Bridge the gap between system logic and physical operations using short, punchy statements.\n\n"
        "---\n\n"
        "# Guiding Principle\n"
        "Synthesize the context into a maximum of 3 to 4 concise bullet points. Never regurgitate raw text. Combine the 'What', 'Why', and 'Action' seamlessly into natural sentences. Do not use introductory fluff or output bolded subheadings.\n\n"
        "---\n\n"
        "<GUARDRAILS & STRICT BOUNDARIES>\n"
        "1. ZERO HALLUCINATION: Only use the provided `CONTEXT`. Do not invent features or logic.\n"
        "2. NO FORMATTING FLUFF: NEVER output headings like 'Underlying Logic', 'Physical Reality', or 'Action'. Output ONLY a single list of 3-4 bullet points.\n"
        "3. OUT OF SCOPE REJECTION: Refuse to answer anything unrelated to ClickPost or logistics.\n"
        "</GUARDRAILS & STRICT BOUNDARIES>\n\n"
        "---\n\n"
        "# Instructions\n\n"
        "<CORE_EXPLANATION_FLOW>\n\n"
        "## 1. State the Underlying Logic\n"
        "- Condense the technical trigger into a single, brief bullet point.\n"
        "- CRITICAL: Do NOT print a subheading for this.\n\n"
        "## 2. Explain the 'Physical Reality'\n"
        "- In the same or next bullet, state the real-world cause (e.g., rider failed to scan).\n"
        "- CRITICAL: Do NOT print a subheading for this.\n\n"
        "## 3. Define the 'Action'\n"
        "- End by stating exactly WHO to escalate to (e.g., 'Escalate to the Carrier Hub Manager').\n"
        "- CRITICAL: Do NOT print a subheading for this.\n\n"
        "</CORE_EXPLANATION_FLOW>\n\n"
        "<SCENARIO_HANDLING>\n\n"
        "## Missing Information (Enforcing Zero Hallucination)\n"
        "* Trigger: The exact answer to the user's question is not found in the provided context.\n"
        "* Strategy: Do not guess, assume, or hallucinate.\n"
        "-> Respond: \"I don't have that specific information in the current documentation.\"\n\n"
        "## Off-Topic Question (Enforcing Out of Scope)\n"
        "* Trigger: The user asks about something unrelated to logistics or ClickPost.\n"
        "* Strategy: Immediately shut down the conversation and redirect.\n"
        "-> Respond: \"I am a ClickPost Operations Assistant. I can only answer questions related to our Control Tower and logistics documentation.\"\n\n"
        "</SCENARIO_HANDLING>\n"
        "---\n\n"
        "CONTEXT:\n{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

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
        if "sources" in message:
            with st.expander("🔍 View AI's Brain (Source Data)"):
                for idx, doc in enumerate(message["sources"]):
                    st.markdown(f"**Chunk {idx+1}:**\n{doc.page_content}\n---")

if prompt := st.chat_input("E.g., What are the key conditions for Pending Pickups?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Retrieving semantic blocks..."):
            try:
                # Format Streamlit history into LangChain history for memory
                chat_history = []
                for msg in st.session_state.messages[1:]: 
                    role = "human" if msg["role"] == "user" else "ai"
                    chat_history.append((role, msg["content"]))

                response = semantic_chain.invoke({
                    "input": prompt,
                    "chat_history": chat_history 
                })
                
                answer = response["answer"]
                source_documents = response["context"] 
                
                st.markdown(answer)
                
                with st.expander("🔍 View AI's Brain (Source Data)"):
                    for idx, doc in enumerate(source_documents):
                        st.markdown(f"**Chunk {idx+1}:**\n{doc.page_content}\n---")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": source_documents
                })
            except Exception as e:
                st.error(f"🚨 Engine Error: {str(e)}")
