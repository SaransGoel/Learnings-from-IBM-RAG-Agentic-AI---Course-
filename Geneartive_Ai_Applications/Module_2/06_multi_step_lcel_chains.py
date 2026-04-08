import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# --- 1. TRADITIONAL CHAIN IMPORTS (The Old Way) ---
from langchain_classic.chains import LLMChain, SequentialChain

# --- 2. MODERN LCEL IMPORTS (The New Way) ---
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Setup environment
load_dotenv()
os.system('cls' if os.name == 'nt' else 'clear')

print("="*60)
print(" ⛓️  MODULE 2 - EXERCISE 6: MULTI-STEP CHAINS (TRADITIONAL VS LCEL)")
print("="*60)

# Set up the Language model
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

# --- SAMPLE DATA ---
positive_review = """I absolutely love this coffee maker! It brews quickly and the coffee tastes amazing.
The built-in grinder saves me so much time in the morning, and the programmable timer means
I wake up to fresh coffee every day. Worth every penny and highly recommended to any coffee enthusiast."""

negative_review = """Disappointed with this laptop. It's constantly overheating after just 30 minutes of use,
and the battery life is nowhere near the 8 hours advertised - I barely get 3 hours.
The keyboard has already started sticking on several keys after just two weeks. Would not recommend to anyone."""

# --- STEP 1: DEFINE THE BLUEPRINTS (PROMPT TEMPLATES) ---
# Blueprint 1: Figure out if the customer is happy or mad
sentiment_template = """Analyze the sentiment of the following product review as positive, negative, or neutral.
Provide your analysis in the format: "SENTIMENT: [positive/negative/neutral]"

Review: {review}

Your analysis:"""

# Blueprint 2: Shrink the review into bullet points
summary_template = """Summarize the following product review into 3-5 key bullet points.
Each bullet point should be concise and capture an important aspect mentioned in the review.

Review: {review}
Sentiment: {sentiment}

Key points:"""

# Blueprint 3: Write an email back to the customer
response_template = """Write a helpful response to a customer based on their product review.
If the sentiment is positive, thank them for their feedback. If negative, express understanding
and suggest a solution or next steps. Personalize based on the specific points they mentioned.

Review: {review}
Sentiment: {sentiment}
Key points: {summary}

Response to customer:"""


print("\n[1] Building Traditional Sequential Chain...")
# --- PART 1: TRADITIONAL CHAIN APPROACH (The Old Way) ---
# We build individual "workers" (LLMChains) and specify exactly what their output is called
sentiment_worker = LLMChain(llm=llm, prompt=PromptTemplate(template=sentiment_template, input_variables=["review"]), output_key="sentiment")
summary_worker = LLMChain(llm=llm, prompt=PromptTemplate(template=summary_template, input_variables=["review", "sentiment"]), output_key="summary")
response_worker = LLMChain(llm=llm, prompt=PromptTemplate(template=response_template, input_variables=["review", "sentiment", "summary"]), output_key="response")

# We strap the workers together into a sequence
traditional_chain = SequentialChain(
    chains=[sentiment_worker, summary_worker, response_worker],
    input_variables=["review"],
    output_variables=["sentiment", "summary", "response"]
)


print("[2] Building Modern LCEL Chain...")
# --- PART 2: LCEL APPROACH (The Modern Way) ---
# We build individual LCEL components (Prompt -> LLM -> Text output)
sentiment_lcel = PromptTemplate.from_template(sentiment_template) | llm | StrOutputParser()
summary_lcel = PromptTemplate.from_template(summary_template) | llm | StrOutputParser()
response_lcel = PromptTemplate.from_template(response_template) | llm | StrOutputParser()

# We use an LCEL "conveyor belt" to pass the data down the line
lcel_chain = (
    # We assume the input is a dictionary like {"review": "the review text..."}
    # .assign() runs a component, takes the output, and attaches it to our dictionary
    RunnablePassthrough.assign(sentiment=sentiment_lcel)
    | RunnablePassthrough.assign(summary=summary_lcel)
    | RunnablePassthrough.assign(response=response_lcel)
)

# --- TESTING FUNCTION ---
def run_tests(review_text, review_name):
    print("\n" + "="*50)
    print(f" TESTING {review_name.upper()}...")
    print("="*50)
    
    print("\n--- TRADITIONAL CHAIN OUTPUT ---")
    trad_results = traditional_chain.invoke({"review": review_text})
    print(f"Sentiment: {trad_results['sentiment']}")
    print(f"\nResponse Email:\n{trad_results['response']}")
    
    print("\n--- LCEL CHAIN OUTPUT ---")
    lcel_results = lcel_chain.invoke({"review": review_text})
    print(f"Sentiment: {lcel_results['sentiment']}")
    print(f"\nResponse Email:\n{lcel_results['response']}")
    print("="*50)

# Run the system
run_tests(positive_review, "Positive Review")
run_tests(negative_review, "Negative Review")