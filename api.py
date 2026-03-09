from fastapi import FastAPI
from pydantic import BaseModel
from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

# 1. Setup the Database Connection
CONNECTION_STRING = "postgresql+psycopg://admin:clickpost123@localhost:5432/ct_copilot"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# 2. Initialize the Vector Store (connecting to the tables you just saw)
vectorstore = PGVector(
    collection_name="control_tower_docs",
    connection=CONNECTION_STRING,
    embeddings=embeddings,
)

# 3. Define the LLM (Gemini)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

class QueryRequest(BaseModel):
    question: str

@app.post("/chat")
def chat_with_copilot(request: QueryRequest):
    # Retrieve the top 3 most relevant chunks from your data.txt
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Phase-1 Prompting: Strict Knowledge Grounding
    system_prompt = (
        "You are the ClickPost Control Tower Copilot. Use the retrieved context to answer. "
        "If you don't know the answer, say you don't know and provide a documentation link."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    # Build and run the chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    response = rag_chain.invoke({"input": request.question})
    
    return {"answer": response["answer"]}