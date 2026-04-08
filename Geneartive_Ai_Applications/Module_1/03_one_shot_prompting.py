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
print(" IBM LAB: MODULE 1 LAB EXERCISE 3 ")
print("="*60)

# --- EXERCISE 3: ONE-SHOT PROMPTS ---
print("\n=== EXERCISE 3: ONE-SHOT PROMPTS ===")
# Task: Simplify Technical Concept
oneshot_prompt = PromptTemplate.from_template("""
Example:
Concept: Quantum Entanglement
Simple: It's like having two magic dice. If one rolls a 6, the other instantly rolls a 6, no matter how far apart they are.

Now do this one:
Concept: {concept}
Simple:"""
)
print(f"One-Shot Concept: {(oneshot_prompt | llm_creative).invoke({'concept': 'Blockchain'}).content}")
