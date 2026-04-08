# 🧠 AI ENGINEERING CHEAT SHEET: MODULE 1 (LANGCHAIN BASICS)

## 1. THE FACTORY ARCHITECTURE (LCEL)
LangChain Expression Language (LCEL) is the modern way to build AI pipelines. It acts like a factory assembly line, using the pipe symbol `|` (the conveyor belt) to pass data from left to right.

**The Standard 3-Step Flow:**
1. **The Blueprint (`PromptTemplate`):** The instruction form with blank spaces for the AI to follow.
2. **The Brain (`ChatGroq`):** The actual Large Language Model (LLM) doing the thinking.
3. **The Conveyor Belt (`|`):** The symbol that snaps the Blueprint directly into the Brain.

```python
# Architecture Syntax
chain = prompt_template | llm

# .invoke() is the "Start Button". .content rips away the messy metadata to give you clean text.
result = chain.invoke({"variable": "user_input"}).content
```

## 2. SECURITY & ENVIRONMENT (THE VAULT)
Never put your API keys directly into your Python code. If you upload it to GitHub, bots will steal your key. Always hide them in a .env file and use the dotenv library to securely load them into your computer's background memory.

```python
import os
from dotenv import load_dotenv

# Silently opens your .env file and loads the keys so LangChain can find them automatically.
load_dotenv()
```
## 3. CONTROLLING THE BRAIN (TEMPERATURE)
When initializing the LLM, the temperature parameter (from 0.0 to 1.0) controls how much the AI is allowed to "hallucinate" or be creative.

1. **Temperature 0.0 (Strict/Robotic):** The AI picks the most statistically likely word every time.Best for coding, math, data extraction, and rigid formatting.

2. **Temperature 0.8+ (Creative):** The AI takes risks and uses varied vocabulary. Best for brainstorming, creative writing, and generating wild ideas.

```python
from langchain_groq import ChatGroq

strict_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
creative_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.8)
```
## 4. PROMPT TEMPLATES (THE BLUEPRINTS)
Templates use curly brackets {} to define dynamic blank spaces (like a Mad Libs game). To actually run the prompt, you pass a Python Dictionary into .invoke() to fill in those exact blanks.
```python
from langchain_core.prompts import PromptTemplate

# 1. Write the string with {} as your blank spaces
blueprint = "Write a {tone} script about {topic}."

# 2. Convert it into an official LangChain machine component
prompt = PromptTemplate.from_template(blueprint)

# 3. Snap it to the LLM and fill the blanks using a dictionary
chain = prompt | creative_llm
chain.invoke({"tone": "dark", "topic": "the ocean"})
```
## 5. PROMPT ENGINEERING TECHNIQUES
The science of writing better instructions so the AI doesn't make mistakes or get confused.

### A. Zero-Shot Prompting
Giving the AI a strict command with zero prior examples. It relies entirely on its pre-trained knowledge.

**Use case:** Simple classification, summarization, or translation.
```python
"Classify this video idea into ONE category: 'Science' or 'Philosophy'. Idea: {idea}"
```
### B. Few-Shot Prompting
Giving the AI 2 to 5 examples of exactly what you want before asking it to do the job.

**Use case:** Forcing the AI to copy a highly specific tone, strict formatting, or brand voice (like cynical "Beyond Obvious" titles).
```python
"""
Write a cynical video title.
Example 1: The concept of time is a corporate lie -> "You Don't Own Your Time (They Do)"
Example 2: Social media algorithms -> "The Algorithm is your Real God"

Now do this one: {new_topic} ->
"""
```
### C. Chain-of-Thought (CoT) Prompting
Forcing the AI to "show its work" and think step-by-step before giving the final answer. This drastically reduces logic errors and hallucinations.

**Use case:** Complex math, deep logical reasoning, or strict architectural outlining.
```python
"""
Break down your thought process step-by-step:
Step 1: Write the hook.
Step 2: Explain the obvious lie.
Step 3: Reveal the dark reality.
"""
```