import time
from google import genai


class FileUploader:
    """Handles uploading files to Gemini File Search Stores."""

    POLL_INTERVAL = 5  # seconds

    def __init__(self, client: genai.Client):
        self.client = client

    def _wait_for_operation(self, operation):
        """Poll an operation until it completes."""
        while not operation.done:
            time.sleep(self.POLL_INTERVAL)
            operation = self.client.operations.get(operation)
        print("[upload] Operation complete")
        return operation

    def upload_direct(
        self,
        file_path: str,
        store_name: str,
        display_name: str,
        custom_metadata: list[dict] | None = None,
        chunking_config: dict | None = None,
    ):
        """
        Directly upload a file into a File Search Store.
        Combines upload + import in one step.
        """
        config = {"display_name": display_name}
        if custom_metadata:
            config["custom_metadata"] = custom_metadata
        if chunking_config:
            config["chunking_config"] = chunking_config

        print(f"[upload] Uploading {file_path} to {store_name}...")
        operation = self.client.file_search_stores.upload_to_file_search_store(
            file=file_path,
            file_search_store_name=store_name,
            config=config,
        )
        return self._wait_for_operation(operation)

    def upload_and_import(
        self,
        file_path: str,
        store_name: str,
        file_display_name: str,
        custom_metadata: list[dict] | None = None,
    ):
        """
        Upload a file via Files API then import it into a File Search Store.
        Useful when you want to reuse the same file across multiple stores.
        """
        print(f"[upload] Uploading {file_path} via Files API...")
        uploaded_file = self.client.files.upload(
            file=file_path,
            config={"name": file_display_name},
        )
        print(f"[upload] File uploaded: {uploaded_file.name}")

        import_config = {}
        if custom_metadata:
            import_config["custom_metadata"] = custom_metadata

        print(f"[upload] Importing {uploaded_file.name} into {store_name}...")
        operation = self.client.file_search_stores.import_file(
            file_search_store_name=store_name,
            file_name=uploaded_file.name,
            **import_config,
        )
        return self._wait_for_operation(operation)

    def upload_with_chunking(
        self,
        file_path: str,
        store_name: str,
        display_name: str,
        max_tokens_per_chunk: int = 200,
        max_overlap_tokens: int = 20,
    ):
        """Upload a file with custom chunking configuration."""
        chunking_config = {
            "white_space_config": {
                "max_tokens_per_chunk": max_tokens_per_chunk,
                "max_overlap_tokens": max_overlap_tokens,
            }
        }
        return self.upload_direct(
            file_path=file_path,
            store_name=store_name,
            display_name=display_name,
            chunking_config=chunking_config,
        )
