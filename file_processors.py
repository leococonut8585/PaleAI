import mimetypes
from typing import Dict

async def stage0_process(filename: str, content: bytes) -> Dict[str, str]:
    """Placeholder Stage 0 processing for uploaded files."""
    ext = filename.split('.')[-1].lower()
    size = len(content)
    # TODO: integrate with actual AI services based on file type
    summary = f"Received {ext} file of {size} bytes."
    return {"summary": summary}
