"""
tqlembed20 — Gemini File Search RAG
Demonstrates the full Gemini File Search API: create stores, upload files,
query with semantic search, retrieve citations, and return structured output.
"""

import os
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src import FileSearchStoreManager, FileUploader, FileSearchQuery

load_dotenv()
console = Console()


def demo_basic_rag(client: genai.Client):
    """Basic RAG: upload a file and query it."""
    console.rule("[bold blue]Basic RAG Demo")

    store_mgr = FileSearchStoreManager(client)
    uploader = FileUploader(client)
    query_engine = FileSearchQuery(client)

    # Create a File Search Store
    store = store_mgr.create(display_name="tqlembed20-demo")

    # Upload sample document directly to the store
    uploader.upload_direct(
        file_path="sample_data/sample.txt",
        store_name=store.name,
        display_name="TQL Guidelines",
    )

    # Query the store
    answer = query_engine.ask(
        question="What is the minimum credit score for the DSCR loan program?",
        store_names=[store.name],
    )
    console.print(Panel(answer, title="[green]Answer", border_style="green"))

    return store.name


def demo_metadata_filter(client: genai.Client, store_name: str):
    """Demonstrate metadata-filtered queries."""
    console.rule("[bold blue]Metadata Filter Demo")

    uploader = FileUploader(client)
    query_engine = FileSearchQuery(client)

    # Upload a file with custom metadata
    uploader.upload_and_import(
        file_path="sample_data/sample.txt",
        store_name=store_name,
        file_display_name="guidelines-with-meta",
        custom_metadata=[
            {"key": "program", "string_value": "DSCR"},
            {"key": "version", "numeric_value": 2026},
        ],
    )

    # Query with metadata filter
    result = query_engine.ask_with_citations(
        question="What are the LTV limits?",
        store_names=[store_name],
        metadata_filter='program = "DSCR"',
    )

    console.print(Panel(result["answer"], title="[green]Answer", border_style="green"))

    if result["citations"]:
        table = Table(title="Citations", show_header=True)
        table.add_column("Title")
        table.add_column("Page")
        table.add_column("Metadata")
        for c in result["citations"]:
            table.add_row(
                c.get("title") or "-",
                str(c.get("page_number") or "-"),
                str(c.get("metadata") or "-"),
            )
        console.print(table)


def demo_structured_output(client: genai.Client, store_name: str):
    """Demonstrate structured JSON output from File Search."""
    console.rule("[bold blue]Structured Output Demo")

    class LoanProgram(BaseModel):
        program_name: str = Field(description="Name of the loan program")
        min_credit_score: int = Field(description="Minimum credit score required")
        max_ltv_purchase: str = Field(description="Maximum LTV for purchase")
        min_loan_amount: str = Field(description="Minimum loan amount")
        max_loan_amount: str = Field(description="Maximum loan amount")

    query_engine = FileSearchQuery(client)
    result = query_engine.ask_structured(
        question="Summarize the Bank Statement loan program requirements.",
        store_names=[store_name],
        schema=LoanProgram,
    )

    console.print(Panel(
        f"Program: [bold]{result.program_name}[/bold]\n"
        f"Min Credit Score: {result.min_credit_score}\n"
        f"Max LTV (Purchase): {result.max_ltv_purchase}\n"
        f"Loan Range: {result.min_loan_amount} – {result.max_loan_amount}",
        title="[green]Structured Result",
        border_style="green",
    ))


def demo_custom_chunking(client: genai.Client, store_name: str):
    """Demonstrate custom chunking configuration."""
    console.rule("[bold blue]Custom Chunking Demo")

    uploader = FileUploader(client)
    uploader.upload_with_chunking(
        file_path="sample_data/sample.txt",
        store_name=store_name,
        display_name="chunked-guidelines",
        max_tokens_per_chunk=150,
        max_overlap_tokens=15,
    )
    console.print("[green]Custom chunking upload complete.[/green]")


def demo_store_management(client: genai.Client):
    """List all stores and their documents."""
    console.rule("[bold blue]Store Management Demo")

    store_mgr = FileSearchStoreManager(client)
    stores = store_mgr.list()

    table = Table(title="File Search Stores", show_header=True)
    table.add_column("Name")
    table.add_column("Display Name")
    for s in stores:
        table.add_row(s.name, getattr(s, "display_name", "-"))
    console.print(table)


def cleanup(client: genai.Client, store_name: str):
    """Delete the demo store."""
    store_mgr = FileSearchStoreManager(client)
    store_mgr.delete(store_name)
    console.print(f"[yellow]Cleaned up store: {store_name}[/yellow]")


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        console.print("[red]Error: GEMINI_API_KEY not set in .env[/red]")
        return

    client = genai.Client(api_key=api_key)

    console.print(Panel.fit(
        "[bold blue]tqlembed20[/bold blue] — Gemini File Search RAG\n"
        "Semantic search over your documents using Gemini embeddings",
        border_style="blue",
    ))

    store_name = demo_basic_rag(client)
    demo_metadata_filter(client, store_name)
    demo_structured_output(client, store_name)
    demo_custom_chunking(client, store_name)
    demo_store_management(client)

    console.print("\n[dim]Run cleanup(client, store_name) to delete the demo store.[/dim]")


if __name__ == "__main__":
    main()
