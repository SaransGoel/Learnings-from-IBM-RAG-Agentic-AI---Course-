# %%
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load secret vault
load_dotenv()

# Setup analytical and creative brains
llm_precise = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)
llm_creative = ChatGroq(model="llama-3.1-8b-instant", temperature=0.8)

print("="*60)
print(" IBM LAB: MODULE 1 LAB EXERCISE 1 ")
print("="*60)


# --- EXERCISE 1: BASIC PROMPTS ---
print("\n=== EXERCISE 1: BASIC PROMPT ===")
basic_prompt = PromptTemplate.from_template("{text}")
basic_chain = basic_prompt | llm_creative
ex1_response = basic_chain.invoke({"text": "The future of artificial intelligence is"}).content
print(f"Completion: {ex1_response}")
