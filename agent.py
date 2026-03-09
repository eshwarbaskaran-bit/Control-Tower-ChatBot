import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# Only needed for the LLM reasoning now!
os.environ["GOOGLE_API_KEY"] = "AIzaSyB9yFvf598dHn_OJ-5VJ-C_VXc-eyGsXok" 

class SemanticSniperAgent:
    def __init__(self):
        # 1. Initialize Local Embeddings (MUST match ingest.py exactly)
        print("🧠 [LOCAL]: Loading HuggingFace Semantic Model...")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        
        # 2. Load the Semantic FAISS Database
        print("💾 [DB]: Loading Semantic Index...")
        self.vector_db = FAISS.load_local(
            "faiss_index", 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )

        # 3. Setup Gemini 2.5 Flash for reasoning
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

        # 4. High-Precision Retriever
        # Because we have 90 semantic chunks, k=4 will pull 4 entire topics.
        self.retriever = self.vector_db.as_retriever(
            search_type="mmr", 
            search_kwargs={'k': 4, 'fetch_k': 20}
        )

        # 5. The Professional Architect Prompt
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
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # 6. Build the Chain
        document_chain = create_stuff_documents_chain(self.llm, prompt_template)
        self.retrieval_chain = create_retrieval_chain(self.retriever, document_chain)

    def ask(self, query):
        response = self.retrieval_chain.invoke({"input": query})
        return response["answer"]

if __name__ == "__main__":
    bot = SemanticSniperAgent()
    
    # Test it with a "logic" question to see the semantic power
    question = "How does pending pickups work?"
    print(f"\n👤 USER: {question}\n")
    
    answer = bot.ask(question)
    print(f"🎯 SEMANTIC RESPONSE:\n{answer}")
