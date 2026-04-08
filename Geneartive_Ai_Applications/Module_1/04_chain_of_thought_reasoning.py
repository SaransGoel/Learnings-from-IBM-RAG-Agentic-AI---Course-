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
print(" IBM LAB: MODULE 1 LAB EXERCISE 4 ")
print("="*60)

# --- EXERCISE 4: CHAIN-OF-THOUGHT (CoT) ---
print("\n=== EXERCISE 4: CHAIN OF THOUGHT ===")
cot_prompt = PromptTemplate.from_template("""
Consider the problem: 'A student has a big test in 2 days. Should they study tonight or go to a movie with friends?'
Break down the decision-making process step-by-step before giving a final recommendation.
""")
print(f"CoT Reasoning:\n{(cot_prompt | llm_precise).invoke({}).content}")

print("\n" + "="*60)
print("✅ MODULE 1 LAB COMPLETE.")
