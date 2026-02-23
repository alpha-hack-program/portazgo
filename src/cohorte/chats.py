# Copyright 2025 IBM, Red Hat
# SPDX-License-Identifier: Apache-2.0

"""
Chat message and stream event types for chatbot use (Streamlit, etc.).
"""

from typing import Any, Dict, List, TypedDict


class ChatMessage(TypedDict, total=False):
    """One message in a conversation. role + content are required."""

    role: str  # "user" | "assistant" | "system"
    content: str


def format_history_as_prefix(messages: List[Dict[str, str]]) -> str:
    """
    Format a list of chat messages as a single string prefix for context.

    Use this when the backend only accepts one input: prepend this to the
    current user input so the model sees conversation history.
    """
    if not messages:
        return ""
    lines = []
    for m in messages:
        role = (m.get("role") or "user").strip().lower()
        content = (m.get("content") or "").strip()
        if role == "system":
            lines.append(f"System: {content}")
        elif role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"{role.capitalize()}: {content}")
    return "\n".join(lines) + "\n\n"


# Stream event types (for invoke_stream)
# - content_delta: {"type": "content_delta", "delta": str}
# - done: {"type": "done", "answer": str, "contexts": list, "tool_calls": list}
StreamEvent = Dict[str, Any]
