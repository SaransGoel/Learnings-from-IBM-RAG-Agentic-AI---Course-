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
print(" IBM LAB: MODULE 1 LAB EXERCISE 2 ")
print("="*60)

# --- EXERCISE 2: ZERO-SHOT PROMPTS ---
print("\n=== EXERCISE 2: ZERO-SHOT PROMPTS ===")
# Task 1: Movie Review
movie_prompt = PromptTemplate.from_template("Classify this movie review as positive or negative: {review}\nCategory:")
print(f"Movie Review: {(movie_prompt | llm_precise).invoke({'review': 'The cinematography was beautiful but the plot was boring.'}).content}")

# Task 2: Climate Summarization
climate_prompt = PromptTemplate.from_template("Summarize this paragraph in one sentence: {text}\nSummary:")
print(f"Climate Summary: {(climate_prompt | llm_precise).invoke({'text': 'Global warming melts ice caps, raising sea levels and threatening cities.'}).content}")

# Task 3: Translation
translate_prompt = PromptTemplate.from_template("Translate this to Spanish: {text}\nTranslation:")
print(f"Translation: {(translate_prompt | llm_precise).invoke({'text': 'Where is the nearest supermarket?'}).content}")
