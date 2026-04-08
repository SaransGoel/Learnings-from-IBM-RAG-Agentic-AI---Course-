# %%
# EXERCISE 1: Comparing Model Temperatures
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# 1. Loading API
load_dotenv()

# 2. Setup Model A: Very robotic and precise (Temperature 0.0)
llm_precise = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0
)

# 3. Setup Model B: Highly creative and unpredictable (Temperature 0.9)
llm_creative = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.9
)

# 4. Define our 3 test prompts required by the lab
prompts = [
    "Write a short, 2-line poem about artificial intelligence.", # Creative
    "What are the key components of a neural network?",          # Factual
    "List 3 quick tips for effective time management."           # Instruction
]

# 5. Run the experiment
for question in prompts:
    print(f"\n--- QUERY: {question} ---")
    
    # Send the question to the strict model
    precise_answer = llm_precise.invoke(question)
    print(f"\n🤖 PRECISE MODEL (Temp 0.0):\n{precise_answer.content}")
    
    # Send the exact same question to the creative model
    creative_answer = llm_creative.invoke(question)
    print(f"\n🎨 CREATIVE MODEL (Temp 0.9):\n{creative_answer.content}")
    print("\n" + "="*50)