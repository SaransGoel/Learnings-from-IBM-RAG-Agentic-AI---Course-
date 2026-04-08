# 📦 PROJECT DEPENDENCIES & TECH STACK

This document outlines the core Python packages used to build the AI applications, content engines, and RAG pipelines in this repository.

## 1. Core AI & Architecture (Module 1 & 2)
* **`langchain`**: The main umbrella framework. We use it for conversational memory (`ConversationBufferMemory`) and complex chains (`ConversationChain`).
* **`langchain-core`**: The foundational building blocks (`PromptTemplate`, `OutputParsers`, and the LCEL `|` syntax).
* **`langchain-groq`**: Connects our pipeline to the lightning-fast Groq Llama-3 models.
* **`python-dotenv`**: Securely loads hidden `.env` files so we don't leak API keys to the public.

## 2. Ingestion & Data Engineering (Module 2 - Exercise 3)
* **`langchain-community`**: A toolbox of integrations (gives us `WebBaseLoader` and `PyPDFLoader`).
* **`beautifulsoup4`**: The web-scraping engine that strips messy HTML into readable text.
* **`pypdf`**: The engine that cracks open PDF files so our shredders can read them.

## 3. Vector Databases & Embeddings (Module 2 - Exercise 4)
* **`chromadb`**: Our local Vector Database. It stores our shredded text chunks as mathematical coordinates so we can search them instantly.
* **`langchain-huggingface`**: Connects LangChain to open-source mathematical models.
* **`sentence-transformers`**: The actual "Mathematician" model that reads our text and converts it into those mathematical coordinates (Embeddings).

## 4. Agents & Tools (Module 2 - Exercise 7)
* **`langchain-experimental`**: Contains advanced tools for AI Agents. We will use this specifically for the `PythonREPL` tool, which allows the AI to write and execute its own Python code to solve math problems!