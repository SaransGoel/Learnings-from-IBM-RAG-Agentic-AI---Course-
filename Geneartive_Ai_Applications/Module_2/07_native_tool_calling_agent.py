import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# Setup environment
load_dotenv()
os.system('cls' if os.name == 'nt' else 'clear')

print("="*60)
print(" 🤖 MODULE 2 - EXERCISE 7: NATIVE TOOL CALLING AGENT (THE MODERN WAY)")
print("="*60)

# --- STEP 1: SET UP THE LLM ---
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)

# --- STEP 2: DEFINE TOOLS USING THE @tool DECORATOR ---
# The @tool decorator automatically translates your Python function 
# into a JSON schema that the LLM API can natively understand!

@tool
def calculator(expression: str) -> str:
    """A simple calculator that can add, subtract, multiply, or divide two numbers.
    Input should be a mathematical expression like '2+2' or '15/3'."""
    try:
        # Note for your OS project: Be careful using eval() with untrusted user input!
        return str(eval(expression))
    except Exception as e:
        return f"Error calculating: {str(e)}"

@tool
def format_text(format_type: str, text: str) -> str:
    """Format text to uppercase, lowercase, or title case."""
    try:
        format_type = format_type.strip().lower()
        content = text.strip()
        
        if format_type == "uppercase": return content.upper()
        elif format_type == "lowercase": return content.lower()
        elif format_type == "titlecase": return content.title()
        else: return "Error: Unknown format type."
    except Exception as e:
        return f"Error formatting text: {str(e)}"
tools = [calculator, format_text]

# --- STEP 3: BUILD THE MODERN PROMPT ---
# Notice how incredibly clean this is compared to the ReAct prompt. 
# We just give it the tools and tell it to get to work.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful system assistant. Use your available tools to answer the user's questions accurately."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# --- STEP 4: ASSEMBLE THE TOOL-CALLING AGENT ---
# This entirely replaces create_react_agent
agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True
)

# --- STEP 5: RUN THE TESTS ---
test_questions = [
    "What is 25 + 63?",
    "Can you convert 'hello world' to uppercase?",
    "Calculate 15 * 7",
    "titlecase: langchain is awesome",
]

for question in test_questions:
    print(f"\n{'='*50}")
    print(f" USER QUERY: {question}")
    print(f"{'='*50}\n")
    try:
        agent_executor.invoke({"input": question})
    except Exception as e:
        print(f"\nAgent hit an error: {e}")

print("\n🎉 EXERCISE 7 COMPLETE! The LLM has successfully used native tool calling.")