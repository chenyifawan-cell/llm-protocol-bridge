import hashlib
import json
from typing import Any, Dict, List, Optional

class StreamTranslationStateMachine:
    """Translates streaming SSE chunks across OpenAI, Anthropic, and Gemini formats."""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.synthetic_id_map: Dict[str, str] = {}
        self.json_buffers: Dict[int, str] = {}

    def synthesize_gemini_tool_id(self, part_index: int, fn_name: str) -> str:
        digest = hashlib.sha256(f"{self.request_id}:{part_index}".encode()).hexdigest()[:10]
        call_id = f"call_{digest}_{part_index}"
        self.synthetic_id_map[call_id] = fn_name
        return call_id

    def buffer_anthropic_delta(self, block_index: int, partial_json: str) -> str:
        current = self.json_buffers.get(block_index, "") + partial_json
        self.json_buffers[block_index] = current
        return partial_json

    def finalize_anthropic_block(self, block_index: int) -> Dict[str, Any]:
        raw = self.json_buffers.pop(block_index, "{}")
        # Ensure balanced brackets/escapes before emitting terminal tool_calls frame
        parsed = json.loads(raw)
        return {"index": block_index, "arguments": json.dumps(parsed), "finish_reason": "tool_calls"}
