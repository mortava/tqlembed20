"""One-off audit: list all File Search Stores and their documents."""
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

stores = list(client.file_search_stores.list())
print(f"Found {len(stores)} store(s):")
for s in stores:
    display = getattr(s, "display_name", None) or "(no display name)"
    print(f"\n  STORE: {s.name}")
    print(f"  display_name: {display}")
    try:
        docs = list(client.file_search_stores.documents.list(parent=s.name))
        print(f"  documents ({len(docs)}):")
        for d in docs:
            d_display = getattr(d, "display_name", None) or "(no display name)"
            print(f"    - {d.name} | display={d_display}")
    except Exception as e:
        print(f"  documents: ERROR - {e}")
