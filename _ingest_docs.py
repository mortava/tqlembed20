"""Batch-upload TQL guideline documents into the pinned Quinn Knowledge Base store.

Usage:
    python _ingest_docs.py <folder_path>
    python _ingest_docs.py <file1> <file2> ...

Idempotent: skips any file whose display_name (= basename) is already in the store.
Supported extensions: .pdf .txt .md .docx .csv .json
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from src.store import FileSearchStoreManager
from src.uploader import FileUploader

SUPPORTED_EXTS = {".pdf", ".txt", ".md", ".docx", ".csv", ".json"}

load_dotenv()

PINNED_STORE_NAME = os.getenv(
    "QUINN_STORE_NAME",
    "fileSearchStores/quinn-knowledge-base-geg6qsemdjxo",
)


def collect_files(args: list[str]) -> list[Path]:
    files: list[Path] = []
    for arg in args:
        p = Path(arg).expanduser().resolve()
        if not p.exists():
            print(f"[skip] not found: {arg}")
            continue
        if p.is_dir():
            for sub in p.rglob("*"):
                if sub.is_file() and sub.suffix.lower() in SUPPORTED_EXTS:
                    files.append(sub)
        elif p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            files.append(p)
        else:
            print(f"[skip] unsupported: {p.name}")
    return files


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set in .env")
        return 1

    client = genai.Client(api_key=api_key)
    mgr = FileSearchStoreManager(client)
    uploader = FileUploader(client)

    print(f"Target store: {PINNED_STORE_NAME}")
    try:
        existing_docs = list(client.file_search_stores.documents.list(parent=PINNED_STORE_NAME))
    except Exception as exc:
        print(f"ERROR: cannot list documents for {PINNED_STORE_NAME}: {exc}")
        return 1

    existing_names = {getattr(d, "display_name", None) for d in existing_docs}
    print(f"Existing documents in store: {len(existing_docs)}")
    for d in existing_docs:
        print(f"  - {getattr(d, 'display_name', d.name)}")

    files = collect_files(sys.argv[1:])
    if not files:
        print("\nNo eligible files found.")
        return 0

    print(f"\nFound {len(files)} eligible file(s) to consider:")
    for f in files:
        print(f"  - {f}")

    uploaded = 0
    skipped = 0
    failed = 0
    for f in files:
        display = f.name
        if display in existing_names:
            print(f"\n[skip] {display} (already in store)")
            skipped += 1
            continue
        print(f"\n[upload] {display}")
        try:
            uploader.upload_direct(
                file_path=str(f),
                store_name=PINNED_STORE_NAME,
                display_name=display,
            )
            uploaded += 1
            existing_names.add(display)
        except Exception as exc:
            print(f"[fail] {display}: {exc}")
            failed += 1

    print("\n=== Summary ===")
    print(f"Uploaded: {uploaded}")
    print(f"Skipped (already present): {skipped}")
    print(f"Failed: {failed}")

    docs_after = list(client.file_search_stores.documents.list(parent=PINNED_STORE_NAME))
    print(f"Total documents in store now: {len(docs_after)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
