import re
import logging
from typing import List, Tuple

from pipecat.frames.frames import Frame, TranscriptionFrame, LLMMessagesFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

logger = logging.getLogger(__name__)

__all__ = [
    "retrieve_chunks",
    "RAGProcessor",
]


# ---------------------------------------------------------------------------
# Chunk retrieval ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def retrieve_chunks(query: str, kb_chunks: List[str], top_k: int = 3) -> List[str]:
    """Return the *top_k* knowledge-base chunks most relevant to *query*.

    Strategy:
    1. Simple keyword overlap on the entire chunk text.
    2. Give a heavy (10×) bonus if the keyword appears in the chunk's first line
       (assumed to be an H2 Markdown heading, e.g. "## Skills").
    3. If no chunk scores >0, fall back to the first *top_k* chunks so that the
       prompt is never empty (avoids LLM confusion).
    """

    if not kb_chunks or not query:
        return []

    q_words = set(re.findall(r"[\w']+", query.lower()))
    scored: List[Tuple[int, str]] = []

    for chunk in kb_chunks:
        lines = chunk.split("\n")
        heading = lines[0].lower() if lines else ""
        content_words = set(re.findall(r"[\w']+", chunk.lower()))
        heading_words = set(re.findall(r"[\w']+", heading))

        score = len(q_words & content_words) + 10 * len(q_words & heading_words)
        if score:
            scored.append((score, chunk))

    if not scored:
        logger.debug("No keyword overlap for '%s'; defaulting to first %d chunks", query, top_k)
        return kb_chunks[:top_k]

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


# ---------------------------------------------------------------------------
# Stateless RAG processor ----------------------------------------------------
# ---------------------------------------------------------------------------

class RAGProcessor(FrameProcessor):
    """A lightweight processor that converts user *TranscriptionFrame*s into the
    message format expected by OpenAI-compatible LLM services, after enriching
    them with RAG context.
    """

    def __init__(self, kb_chunks: List[str], *, top_k: int = 3):
        super().__init__()
        self._kb_chunks = kb_chunks
        self._top_k = top_k

    async def process_frame(self, frame: Frame, direction: FrameDirection):  # type: ignore[override]
        # First let the base class perform lifecycle bookkeeping (StartFrame, metrics, etc.)
        await super().process_frame(frame, direction)

        # Only transform user speech transcriptions.
        if not isinstance(frame, TranscriptionFrame):
            await self.push_frame(frame, direction)
            return

        user_query = frame.text
        chunks = retrieve_chunks(user_query, self._kb_chunks, self._top_k)
        context_str = "\n\n---\n\n".join(chunks)
        prompt = f"Context:\n{context_str}\n\n---\nQuestion: {user_query}"
        await self.push_frame(LLMMessagesFrame(messages=[{"role": "user", "content": prompt}]))
