# %%
import os
# The Loaders (The Unboxers)
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
# The Splitters (The Shredders)
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

os.system('cls' if os.name == 'nt' else 'clear')

print("="*60)
print(" 📄 MODULE 2 - EXERCISE 3: LOADERS & SPLITTERS")
print("="*60)

# --- THE LOADERS ---
print("\n[1] Downloading and Loading Documents...")

# 1. Load an academic PDF straight from the internet
pdf_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/96-FDF8f7coh0ooim7NyEQ/langchain-paper.pdf"
pdf_loader = PyPDFLoader(pdf_url)
pdf_document = pdf_loader.load()
print(f"✅ PDF Loaded: Contains {len(pdf_document)} pages.")

# 2. Load a Website
web_url = "https://python.langchain.com/docs/introduction/"
web_loader = WebBaseLoader(web_url)
web_document = web_loader.load()
print(f"✅ Website Loaded successfully.")


# --- THE SPLITTERS ---
print("\n[2] Shredding the PDF into manageable chunks...")

# Splitter A: Basic Character Splitter (Cuts exactly at 300 characters)
splitter_1 = CharacterTextSplitter(chunk_size=300, chunk_overlap=30, separator="\n")

# Splitter B: Recursive Splitter (Smarter: tries to keep paragraphs and sentences together)
splitter_2 = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

# Apply the shredders to our PDF document
chunks_1 = splitter_1.split_documents(pdf_document)
chunks_2 = splitter_2.split_documents(pdf_document)


# --- ANALYZE THE RESULTS ---
# Let's write a quick function to see exactly what the shredder did
def display_document_stats(docs, name):
    total_chunks = len(docs)
    print(f"\n=== {name} Statistics ===")
    print(f"Total number of chunks created: {total_chunks}")
    
    if total_chunks > 0:
        # Show a preview of the very first chunk it created
        print(f"Preview of Chunk #1:\n'{docs[0].page_content[:150]}...'")
        # Show the metadata (like the page number it came from)
        print(f"Metadata preserved: {list(docs[0].metadata.keys())}")

display_document_stats(chunks_1, "Basic Character Splitter")
display_document_stats(chunks_2, "Recursive Character Splitter")

print("\n" + "="*60)