# %%
import os
import random
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Initialize our creative brain and our analytical brain
creative_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.8)
strict_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)

# Clear the terminal for a clean start (works on Windows/Mac/Linux)
os.system('cls' if os.name == 'nt' else 'clear')

print("="*60)
print(" 🎬 BEYOND OBVIOUS: AUTOMATED CONTENT ENGINE (MODULE 1)")
print("="*60)

# --- PAUSE 1 ---
input("\n👉 Press [ENTER] to brainstorm a new 'What If' theory...")

# --- EXERCISE 1: BASIC PROMPTING (Idea Generation) ---
print("\n[1] Brainstorming...")

# We ask the user for a topic to make the app dynamic!
user_input = input("💡 Enter a topic for your video (or press [ENTER] for a random one): ")

# If the user just hits Enter, we pick a random dark concept to break the API cache
if user_input.strip() == "":
    concept_list = [
        "What if the concept of 'free will' was engineered by an AI to keep us...",
        "What if the moon is actually an ancient surveillance device watching...",
        "What if deja vu is actually just our universe's simulation resetting...",
        "What if human DNA contains a hidden countdown timer placed by..."
    ]
    idea_seed = random.choice(concept_list)
else:
    idea_seed = f"What if {user_input}..."

basic_chain = PromptTemplate.from_template("{seed}") | creative_llm
raw_idea = basic_chain.invoke({"seed": idea_seed}).content
full_concept = idea_seed + " " + raw_idea

print(f"RAW IDEA: {full_concept[:250]}...\n")


# --- PAUSE 2 ---
input("👉 Press [ENTER] to classify this theme...")

# --- EXERCISE 2: ZERO-SHOT PROMPTING (Classification) ---
print("\n[2] Classifying the Theme...")
classifier_prompt = PromptTemplate.from_template(
    "You are a strict content categorizer. Read the following video concept and classify it into EXACTLY ONE of these three categories: 'Simulation Theory', 'Societal Control', or 'Cosmic Nihilism'. Return ONLY the category name.\n\nConcept: {concept}\n\nCategory:"
)

classification_chain = classifier_prompt | strict_llm
theme = classification_chain.invoke({"concept": full_concept}).content
print(f"THEME CLASSIFICATION: {theme}\n")


# --- PAUSE 3 ---
input("👉 Press [ENTER] to generate a click-worthy title...")

# --- EXERCISE 3 & 4: FEW-SHOT PROMPTING (Title Generation) ---
print("\n[3] Generating Title...")
few_shot_prompt = PromptTemplate.from_template("""
You are a YouTube strategist for the channel 'Beyond Obvious'. Generate EXACTLY ONE highly cynical, thought-provoking video title based on the given concept. Do not provide a list.

Example 1:
Concept: The government uses social media algorithms to predict and prevent protests before they happen.
Title: "You Don't Have Free Will (And The Algorithm Proves It)"

Example 2:
Concept: Human DNA has a built-in expiration date programmed by an ancient advanced civilization.
Title: "The Terrifying Truth Hidden in Your DNA"

Now, generate EXACTLY ONE title for this new concept:
Concept: {concept}
Title:"""
)

title_chain = few_shot_prompt | creative_llm
youtube_title = title_chain.invoke({"concept": full_concept}).content
print(f"YOUTUBE TITLE: {youtube_title}\n")


# --- PAUSE 4 ---
input("👉 Press [ENTER] to outline the script architecture...")

# --- EXERCISE 5: CHAIN-OF-THOUGHT PROMPTING (Script Outliner) ---
print("\n[4] Outlining the Script (Step-by-Step)...")
cot_prompt = PromptTemplate.from_template("""
You are outlining a 5-minute video script for the topic: {title}. 
Do not write the actual dialogue or script. Only write a strict 4-point architectural outline. Be concise and do not repeat yourself.

Step 1: Write a 1-sentence hook that challenges a common belief.
Step 2: Explain the "Obvious Lie" (what society wants us to believe).
Step 3: Reveal the "Dark Reality" based on the core concept.
Step 4: Provide a cynical concluding thought.

Format your output EXACTLY as those 4 steps.
""")

cot_chain = cot_prompt | strict_llm
script_outline = cot_chain.invoke({"title": youtube_title}).content
print("\n" + script_outline)
print("\n" + "="*60)
print("✅ MODULE 1 PIPELINE COMPLETE.")