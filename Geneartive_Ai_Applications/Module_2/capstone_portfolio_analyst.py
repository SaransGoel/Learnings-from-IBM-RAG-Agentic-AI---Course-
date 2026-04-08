import os
import json
from dotenv import load_dotenv

# 1. Models & Prompts
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage # <-- NEW: Core message types

# 2. Output Parsers & Chains
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field # <-- Native modern Pydantic

# 3. RAG: Loaders, Splitters, Embeddings, Vector Stores
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 4. Agents (Notice Memory is completely gone!)
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

# --- SETUP ---
load_dotenv()
os.system('cls' if os.name == 'nt' else 'clear')

print("="*70)
print(" 📊 QUANTITATIVE PORTFOLIO ENGINE (MODULE 2 CAPSTONE)")
print("="*70)

# Initialize the analytical brain 
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)

# --- PHASE 1: BUILD THE RAG KNOWLEDGE BASE ---
print("\n[1] Ingesting Financial Theory... (Scraping 'Modern Portfolio Theory')")
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Asset_allocation")
raw_docs = loader.load()

# Shred the data 
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(raw_docs)

# Embed and Store
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma.from_documents(documents=chunks, embedding=embedding_model)
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

print("    ✓ Financial context embedded and stored in ChromaDB.")


# --- PHASE 2: DEFINE THE ACTUARIAL & SYSTEM TOOLS ---
print("\n[2] Equipping Agent with Quantitative Tools...")

@tool
def search_financial_theory(query: str) -> str:
    """Use this to search the RAG database for academic definitions regarding risk, return, and portfolio theory."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

@tool
def calculate_annuity_pv(annual_payment: float, rate_percent: float, years: int) -> str:
    """Calculate the Present Value of an ordinary annuity. Useful for CM1 calculations.
    Example input: annual_payment=50000, rate_percent=5.5, years=10."""
    try:
        r = rate_percent / 100.0
        pv = annual_payment * ((1 - (1 + r)**-years) / r)
        return f"The Present Value of the annuity is approximately ₹{pv:,.2f}"
    except Exception as e:
        return f"Calculation error: {e}"

class ClientPortfolioReport(BaseModel):
    client_alias: str = Field(description="A generic alias for the retail client")
    risk_profile: str = Field(description="Categorized as Conservative, Moderate, or Aggressive")
    equity_allocation_pct: int = Field(description="Recommended percentage allocation to equities")
    debt_allocation_pct: int = Field(description="Recommended percentage allocation to debt/fixed income")
    actuarial_rationale: str = Field(description="A brief justification based on risk appetite and return requirements")

@tool
def generate_structured_report(client_scenario: str) -> str:
    """Use this tool ONLY when asked to 'generate a portfolio report' or 'draft a client summary'."""
    parser = JsonOutputParser(pydantic_object=ClientPortfolioReport)
    prompt = ChatPromptTemplate.from_template(
        "You are a quantitative analyst. Generate a portfolio allocation report for the following client scenario: {scenario}\n\n{format_instructions}"
    )
    chain = prompt | llm | parser
    result = chain.invoke({
        "scenario": client_scenario,
        "format_instructions": parser.get_format_instructions()
    })
    return json.dumps(result, indent=2)

tools = [search_financial_theory, calculate_annuity_pv, generate_structured_report]


# --- PHASE 3: ASSEMBLE THE AGENT ---
print("[3] Booting up the Analytical Core...")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an elite quantitative AI assistant designed to support an Authorised Person in managing retail client portfolios. You specialize in applying CM1 financial mathematics principles and assessing risk. Use your tools to evaluate data, run calculations, and generate structured client reports."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Build the Agent Executor (Notice we removed the memory parameter entirely!)
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    max_iterations=3  # <-- THE CIRCUIT BREAKER
)

# --- PHASE 4: THE INTERACTIVE LOOP (MANUAL STATE MANAGEMENT) ---
os.system('cls' if os.name == 'nt' else 'clear')
print("="*60)
print(" 📈 PORTFOLIO MANAGEMENT COMM-LINK ESTABLISHED")
print(" Type 'exit' to shut down.")
print("="*60)

# THE FIX: We manually manage the memory state here!
chat_history = [] 

while True:
    user_input = input("\n[YOU]: ")
    if user_input.lower() in ['exit', 'quit']:
        print("\n[AGENT]: Closing session. Markets never sleep, but we do.")
        break
        
    try:
        # We pass our manual list directly into the executor
        response = agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history 
        })
        
        print(f"\n[AGENT]:\n{response['output']}")
        
        # We append the conversation to our list so it remembers it for the next loop!
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response['output']))
        
    except Exception as e:
        print(f"\n[SYSTEM ERROR]: {e}")