import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# --- The Modern Core Imports ---
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from pydantic import BaseModel, Field

# Load API Key from your root .env file
load_dotenv()

# 1. Define the Strict JSON Schema (What the AI MUST return)
class AIResponse(BaseModel):
    summary: str = Field(description="A one-sentence summary of the user's message")
    sentiment: int = Field(description="Sentiment score from 0 (negative) to 100 (positive)")
    category: str = Field(description="Category of inquiry (e.g., billing, technical, general, feedback)")
    action: str = Field(description="Recommended action for the human support rep to take")
    response: str = Field(description="The actual helpful response to send to the user")

json_parser = JsonOutputParser(pydantic_object=AIResponse)

# 2. Initialize Groq Models (Matching the Lab's dropdown options)
# Groq hosts Llama and Mixtral natively! We will use Gemma to replace IBM Granite.
models = {
    "llama": ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2),
    "mistral": ChatGroq(model="mixtral-8x7b-32768", temperature=0.2),
    "granite": ChatGroq(model="gemma2-9b-it", temperature=0.2) # Stand-in for Granite
}

# 3. The Master Prompt Template
template = """You are an elite AI assistant routing customer support tickets.
{system_prompt}

USER MESSAGE: {user_prompt}

{format_prompt}"""

prompt = PromptTemplate(
    template=template,
    input_variables=["system_prompt", "user_prompt"],
    partial_variables={"format_prompt": json_parser.get_format_instructions()}
)

# 4. The Execution Chain
def get_ai_response(model_name: str, system_prompt: str, user_prompt: str):
    """Routes the prompt to the selected Groq model and parses the JSON."""
    selected_llm = models.get(model_name, models["llama"])
    
    chain = prompt | selected_llm | json_parser
    
    # Execute the chain
    return chain.invoke({
        'system_prompt': system_prompt,
        'user_prompt': user_prompt
    })