"""One-off: delete the old Quinn Default store and create 'Quinn Knowledge Base'."""
import os
from dotenv import load_dotenv
from google import genai

from src.store import FileSearchStoreManager

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
mgr = FileSearchStoreManager(client)

print("Before:")
for s in mgr.list():
    print(f"  - {s.name} | display={getattr(s, 'display_name', None)}")

print("\nDeleting old stores...")
for s in mgr.list():
    mgr.delete(s.name, force=True)

print("\nCreating new store: 'Quinn Knowledge Base'")
new_store = mgr.create("Quinn Knowledge Base")

print("\nAfter:")
for s in mgr.list():
    print(f"  - {s.name} | display={getattr(s, 'display_name', None)}")
    docs = mgr.list_documents(s.name)
    print(f"    documents: {len(docs)}")

print(f"\nNew store ready: {new_store.name}")
