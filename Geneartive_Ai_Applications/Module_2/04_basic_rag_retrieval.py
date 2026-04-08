# %%
import os
from dotenv import load_dotenv

# 1. Loaders and Splitters (From Exercise 3)
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 2. NEW: Embeddings and Vector Store
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 3. Core AI Tools
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Setup environment
load_dotenv()
os.system('cls' if os.name == 'nt' else 'clear')
strict_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)

print("="*60)
print(" 🧠 MODULE 2 - EXERCISE 4: RAG & VECTOR DATABASES")
print("="*60)

# --- STEP 1: INGESTION (Load & Shred) ---
print("[1] Scraping website and shredding into chunks...")
loader = WebBaseLoader("https://python.langchain.com/v0.2/docs/introduction/")
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(loader.load())


# --- STEP 2: EMBEDDING & STORAGE ---
print("[2] Converting text to math and storing in ChromaDB...")
# We use a free, open-source mathematician model from HuggingFace
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# We create the database and shove all our chunks and the math model inside it
vector_database = Chroma.from_documents(documents=chunks, embedding=embedding_model)

# We create a "Retriever" (A librarian whose only job is to search the database)
retriever = vector_database.as_retriever(search_kwargs={"k": 3}) # k=3 means "bring me the top 3 closest chunks"


# --- STEP 3: THE RAG FACTORY LINE ---
print("[3] Assembling the QA Conveyor Belt...\n")

# The Blueprint: Notice we force the AI to ONLY use the provided context!
template = """
You are a helpful assistant. Answer the user's question based ONLY on the following context.
If you don't know the answer based on the context, say "I don't know."

Context: {context}

Question: {question}

Answer:"""
prompt = PromptTemplate.from_template(template)

# The Conveyor Belt (LCEL)
# 1. We take the user's question and pass it to the Retriever to get the context.
# 2. We pass both the context and the question into the Prompt.
# 3. We send it to the LLM.
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt 
    | strict_llm 
    | StrOutputParser()
)

# --- STEP 4: EXECUTION ---
user_question = "What is LangChain?"
print(f"👤 USER: {user_question}")

# We push the Start button!
final_answer = rag_chain.invoke(user_question)

print(f"🤖 AI: {final_answer}")
print("\n" + "="*60)