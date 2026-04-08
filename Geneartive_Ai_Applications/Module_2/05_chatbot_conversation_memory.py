# %%
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# --- THE MODERN V0.3 MEMORY TOOLS ---
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Setup environment
load_dotenv()
os.system('cls' if os.name == 'nt' else 'clear')

print("="*60)
print(" 🧠 MODULE 2 - EXERCISE 5: MODERN CONVERSATIONAL MEMORY")
print("="*60)

# 1. Set up the Language model
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)

# 2. Set up the modern memory storage
# Instead of a single hidden notepad, we create a dictionary to store memory for multiple users!
memory_vault = {}

def get_memory(session_id: str):
    # If this user doesn't have a notepad yet, create a blank one
    if session_id not in memory_vault:
        memory_vault[session_id] = InMemoryChatMessageHistory()
    return memory_vault[session_id]

# 3. Create the Blueprint (Prompt Template)
# Notice how we use MessagesPlaceholder to inject the memory right before the human's new question
blueprint = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])

# 4. The Conveyor Belt (LCEL)
basic_chain = blueprint | llm

# 5. Wrap it in the Memory Manager
# This automatically grabs the history, injects it into the placeholder, and saves the new answer.
chat_bot = RunnableWithMessageHistory(
    basic_chain,
    get_memory,
    input_messages_key="user_input",
    history_messages_key="chat_history"
)

# 6. Function to simulate a conversation
def chat_simulation(bot, inputs, session_id="user_123"):
    print("\n=== Beginning Chat Simulation ===")
    
    for i, user_input in enumerate(inputs):
        print(f"\n--- Turn {i+1} ---")
        print(f"👤 Human: {user_input}")
        
        # We must pass a session_id so the bot knows WHICH notepad to pull!
        response = bot.invoke(
            {"user_input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        
        print(f"🤖 AI: {response.content}")
        
    print("\n=== End of Chat Simulation ===")

# 7. Test with a series of related questions
test_inputs = [
    "Hello, my name is Alice and my favorite color is blue.",
    "I enjoy hiking in the mountains.",
    "Based on my hobbies, what weekend activities would you recommend?",
    "Can you remember both my name and my favorite color?"
]

# Run the simulation!
chat_simulation(chat_bot, test_inputs)

# 8. Examine the raw memory contents
print("\n" + "="*60)
print("📝 PEEKING AT THE RAW MEMORY TRANSCRIPT:")
# We can pull Alice's exact transcript by using her session_id!
print(memory_vault["user_123"].messages)
print("="*60)