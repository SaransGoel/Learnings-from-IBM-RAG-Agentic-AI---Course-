# 🧠 AI ENGINEERING CHEAT SHEET: MODULE 2 (RAG & AGENTIC AI)

## 1. THE KNOWLEDGE WAREHOUSE (RAG)
Retrieval-Augmented Generation (RAG) is the external storage system for your factory. It allows the AI to "look up" manuals or data it wasn't born with.

**The 4-Step Conveyor Belt:**
1. **The Scraper (`Loaders`):** Ingesting raw materials (Web pages, PDFs, CSVs).
2. **The Shredder (`Splitters`):** Breaking documents into "chunks" so the AI doesn't choke on data limits.
3. **The Translator (`Embeddings`):** Converting text into vectors (mathematical coordinates).
4. **The Library (`VectorStores`):** Storing vectors in a database (like ChromaDB) for instant searching.

```python
# The Standard RAG Assembly
loader = WebBaseLoader("https://...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Snaps the chunks and embeddings into the database
vector_db = Chroma.from_documents(documents=chunks, embedding=HuggingFaceEmbeddings())
```

## 2. THE DATA PRESS (OUTPUT PARSERS)
In an automated factory, human sentences are "dirty data." You need JSON to ensure the output fits perfectly into your other system components.

**The Precision Molding:**

1. **The Blueprint (`Pydantic`):** Defining the exact shape, keys, and data types (int, str) the AI must use.
2. **The Hard-Constraint (`JsonOutputParser`):** Forcing the LLM to output only the valid code block, no "fluff" text.
```python
# Defining the strict JSON schema
class SystemUpdate(BaseModel):
    version: float = Field(description="The version number")
    status: str = Field(description="System health status")

parser = JsonOutputParser(pydantic_object=SystemUpdate)

# Snap it into the conveyor belt
chain = prompt | llm | parser
```
## 3. THE ROBOTIC ARMS (NATIVE TOOLS)
Tools are the "hands" of your AI. Without them, the AI can only think. With them, it can execute Python code, run calculators, or check system logs.

**The Mechanical Specs:**

1. **The Trigger (`@tool`):** The AI reads the Docstring (the description) to decide when to use the tool.
2. **The Payload (`Native Parameters`):** Modern agents pass clean JSON arguments directly into your Python function.
```python
# The decorator turns a standard function into an AI tool
@tool
def calculate_depreciation(value: float, years: int):
    """Calculates asset depreciation. Use this for portfolio analysis."""
    return value * (0.9 ** years)
```
## 4. MISSION CONTROL (AGENT EXECUTORS)
An Agent is the autonomous factory supervisor. It runs a loop: Thought → Action → Observation. It doesn't stop until the goal is achieved.

**The Control Panel:**

1. **The Safety Brake (`max_iterations`):** Prevents the agent from getting stuck in an infinite loop.
2. **X-Ray Vision (`verbose=True`):** Essential for debugging. It shows you the agent's internal "thinking" in the terminal.
```python
# Initializing the autonomous loop
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    max_iterations=3 # Circuit breaker
)
```
## 5. THE SHIFT LOGBOOK (MANUAL MEMORY)
For a professional OS, we don't use "automatic" memory. We manage the Chat History manually to ensure the AI only remembers what is relevant to the current task.

**The Record Keeping:**

1. **Message Types (`HumanMessage / AIMessage`):** Categorizing text based on who said it.
2. **State Management (`List`):** Appending each turn to a simple Python list to maintain the context window.
```python
# The Manual Logbook
chat_history = [] 

# Passed into the executor for each turn
response = agent_executor.invoke({
    "input": user_input, 
    "chat_history": chat_history
})
```