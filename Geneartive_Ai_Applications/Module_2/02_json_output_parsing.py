# %%
import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# --- NEW TOOLS FOR MODULE 2 ---
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# 1. Load the vault and wake up the strict brain
load_dotenv()
os.system('cls' if os.name == 'nt' else 'clear')
strict_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)

print("="*60)
print(" 🤖 JSON DATA EXTRACTOR (MODULE 2)")
print("="*60)

# --- THE MOLD (PYDANTIC) ---
# We define EXACTLY what data we want the AI to give us back. 
# It is no longer allowed to just write paragraphs.
class VideoConcept(BaseModel):
    video_title: str = Field(description="A highly cynical, click-worthy YouTube title")
    theme_category: str = Field(description="The core theme (e.g., 'Simulation Theory' or 'Existential Dread')")
    viral_hook: str = Field(description="A one-sentence engaging hook for the intro")

# We hand our mold over to the LangChain Output Parser
json_parser = JsonOutputParser(pydantic_object=VideoConcept)

# --- THE BLUEPRINT ---
# Notice the new variable: {format_instructions}
# The parser automatically writes the strict JSON rules and injects them here!
blueprint = """
You are a YouTube strategist. Generate a video concept based on this seed: {seed}.

{format_instructions}
"""

prompt_form = PromptTemplate(
    template=blueprint,
    input_variables=["seed"],
    partial_variables={"format_instructions": json_parser.get_format_instructions()},
)

# --- THE CONVEYOR BELT ---
# Prompt -> LLM -> JSON Parser
json_chain = prompt_form | strict_llm | json_parser

# --- EXECUTION ---
idea_seed = input("\n💡 Enter a 'What If' topic (e.g., 'Sleep is a software update'): ")

print("\n⚙️ Generating strict JSON data...\n")

# Run the chain!
final_data = json_chain.invoke({"seed": idea_seed})

# --- THE RESULT ---
# We use Python's built-in json library to print the dictionary nicely
print(json.dumps(final_data, indent=4))

# To prove this is usable computer data, we can call specific keys!
print("\n" + "-"*40)
print(f"Isolated Title String: {final_data['video_title']}")
print("-"*40 + "\n")