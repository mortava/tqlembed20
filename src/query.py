from google import genai
from google.genai import types
from pydantic import BaseModel


class FileSearchQuery:
    """Query a Gemini File Search Store using RAG."""

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, client: genai.Client, model: str = DEFAULT_MODEL):
        self.client = client
        self.model = model

    def _build_tool(self, store_names: list[str], metadata_filter: str | None = None):
        file_search_kwargs = {"file_search_store_names": store_names}
        if metadata_filter:
            file_search_kwargs["metadata_filter"] = metadata_filter
        return types.Tool(file_search=types.FileSearch(**file_search_kwargs))

    def ask(
        self,
        question: str,
        store_names: list[str],
        metadata_filter: str | None = None,
    ) -> str:
        """Query the file search store with a question."""
        tool = self._build_tool(store_names, metadata_filter)
        response = self.client.models.generate_content(
            model=self.model,
            contents=question,
            config=types.GenerateContentConfig(tools=[tool]),
        )
        return response.text

    def ask_with_citations(
        self,
        question: str,
        store_names: list[str],
        metadata_filter: str | None = None,
    ) -> dict:
        """Query and return both the answer and citation metadata."""
        tool = self._build_tool(store_names, metadata_filter)
        response = self.client.models.generate_content(
            model=self.model,
            contents=question,
            config=types.GenerateContentConfig(tools=[tool]),
        )

        citations = []
        candidate = response.candidates[0]
        if candidate.grounding_metadata:
            for chunk in candidate.grounding_metadata.grounding_chunks:
                if chunk.retrieved_context:
                    ctx = chunk.retrieved_context
                    citation = {
                        "text": ctx.text if hasattr(ctx, "text") else None,
                        "title": ctx.title if hasattr(ctx, "title") else None,
                        "page_number": ctx.page_number if hasattr(ctx, "page_number") else None,
                        "media_id": ctx.media_id if hasattr(ctx, "media_id") else None,
                    }
                    if hasattr(ctx, "custom_metadata") and ctx.custom_metadata:
                        citation["metadata"] = {
                            m.key: (m.string_value if hasattr(m, "string_value") else m.numeric_value)
                            for m in ctx.custom_metadata
                        }
                    citations.append(citation)

        return {"answer": response.text, "citations": citations}

    def ask_structured(
        self,
        question: str,
        store_names: list[str],
        schema: type[BaseModel],
        metadata_filter: str | None = None,
    ):
        """Query and return a structured Pydantic model response."""
        tool = self._build_tool(store_names, metadata_filter)
        response = self.client.models.generate_content(
            model=self.model,
            contents=question,
            config=types.GenerateContentConfig(
                tools=[tool],
                response_mime_type="application/json",
                response_schema=schema.model_json_schema(),
            ),
        )
        return schema.model_validate_json(response.text)
