# Control-Tower-ChatBot

# ClickPost Control Tower AI 🗼

An Enterprise RAG (Retrieval-Augmented Generation) chatbot designed to act as a Senior Product Operations Analyst for ClickPost. 

This AI ingests internal Product Requirements Documents (PRDs) and operational logic, allowing team members to ask complex logistics questions and receive highly accurate, actionable answers in seconds.



## ✨ Key Features

* **"Sniper" Precision Prompting:** The AI is strictly constrained to answer in a maximum of 3-4 bullet points. It strips out corporate fluff and instantly delivers the system logic, the physical reality (the "Why"), and the exact escalation path.
* **Contextual Memory:** Features a History-Aware Retriever. The bot remembers previous messages, allowing users to ask natural follow-up questions without losing context.
* **Zero Hallucination Guardrails:** If the answer is not in the ingested documentation, the AI is programmed to explicitly refuse to answer rather than guess.
* **Source Transparency:** Every response includes a "View AI's Brain" expander, allowing users to audit the exact semantic text blocks the AI read to formulate its answer.
* **100% Local Embeddings:** Uses local HuggingFace models for document vectorization, meaning company documentation is never sent to a third-party API for embedding.

## 🛠️ Architecture & Tech Stack

* **Frontend UI:** [Streamlit](https://streamlit.io/)
* **Orchestration:** [LangChain](https://www.langchain.com/) (`langchain_classic`)
* **Vector Database:** [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search)
* **Embedding Model:** HuggingFace `sentence-transformers/all-mpnet-base-v2` (Runs locally)
* **Reasoning Engine (LLM):** Google Gemini 2.5 Flash

## 🚀 Local Setup & Installation

**1. Clone the repository**
```bash
git clone [https://github.com/YOUR-USERNAME/control-tower-chatbot.git](https://github.com/YOUR-USERNAME/control-tower-chatbot.git)
cd control-tower-chatbot
