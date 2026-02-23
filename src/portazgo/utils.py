# Copyright 2025 IBM, Red Hat
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for agent responses (Llama Stack–compatible)."""

import json
import re
from typing import Any, Dict, List


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from model output so the stored answer is clean."""
    if not text or not isinstance(text, str):
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def serialize_for_json(val: Any) -> Any:
    """Convert a value to something JSON-serializable (str, dict, list, number, bool, None)."""
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, (dict, list)):
        return val
    if hasattr(val, "__dict__"):
        return {k: serialize_for_json(v) for k, v in val.__dict__.items()}
    return str(val)


def extract_tool_calls(response: Any) -> List[Dict[str, Any]]:
    """
    Extract tool calls from Llama Stack response.

    Supports response.output list (file_search_call, mcp_call, ...), then
    response.turns (tool_calls, steps), then response.tool_calls.

    Returns list of {"tool_name", "arguments", "response"}.
    """
    out: List[Dict[str, Any]] = []

    if hasattr(response, "output") and isinstance(response.output, list):
        for item in response.output:
            type_ = getattr(item, "type", None)
            if type_ == "file_search_call":
                queries = getattr(item, "queries", None) or []
                results = getattr(item, "results", None) or []
                result_texts = [getattr(r, "text", "") or "" for r in results]
                out.append({
                    "tool_name": "file_search",
                    "arguments": {"queries": list(queries)},
                    "response": result_texts,
                })
            elif type_ and "mcp" in str(type_).lower() and type_ != "mcp_list_tools":
                name = (
                    getattr(item, "tool_name", None)
                    or getattr(item, "name", None)
                    or getattr(item, "server_label", None)
                    or "mcp"
                )
                args = getattr(item, "arguments", None) or getattr(item, "args", None) or {}
                resp = (
                    getattr(item, "output", None)
                    or getattr(item, "response", None)
                    or getattr(item, "result", None)
                )
                if not isinstance(args, dict):
                    try:
                        args = json.loads(args) if isinstance(args, str) else {}
                    except Exception:
                        args = {} if args is None else {"raw": str(args)}
                out.append({
                    "tool_name": str(name),
                    "arguments": args,
                    "response": serialize_for_json(resp),
                })
        if out:
            return out

    if hasattr(response, "turns") and response.turns:
        for turn in response.turns:
            if hasattr(turn, "tool_calls") and turn.tool_calls:
                for tc in turn.tool_calls:
                    name = getattr(tc, "tool_name", None) or (
                        tc.__dict__.get("tool_name") if hasattr(tc, "__dict__") else "unknown"
                    )
                    args = getattr(tc, "arguments", None) or (
                        tc.__dict__.get("arguments") if hasattr(tc, "__dict__") else {}
                    )
                    resp = getattr(tc, "response", None) or (
                        tc.__dict__.get("response") if hasattr(tc, "__dict__") else None
                    )
                    if not isinstance(args, dict):
                        try:
                            args = json.loads(args) if isinstance(args, str) else (args or {})
                        except Exception:
                            args = {} if args is None else {"raw": str(args)}
                    out.append({
                        "tool_name": name,
                        "arguments": args,
                        "response": serialize_for_json(resp),
                    })
            if hasattr(turn, "steps") and turn.steps:
                for step in turn.steps:
                    name = getattr(step, "tool_name", None) or (
                        getattr(step, "__dict__", {}).get("tool_name") or "unknown"
                    )
                    args = getattr(step, "tool_args", None) or (
                        getattr(step, "__dict__", {}).get("tool_args") or {}
                    )
                    resp = getattr(step, "tool_response", None) or (
                        getattr(step, "__dict__", {}).get("tool_response")
                    )
                    if not isinstance(args, dict):
                        try:
                            args = json.loads(args) if isinstance(args, str) else {}
                        except Exception:
                            args = {} if args is None else {"raw": str(args)}
                    out.append({
                        "tool_name": name,
                        "arguments": args,
                        "response": serialize_for_json(resp),
                    })
    elif hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            d = tc.__dict__ if hasattr(tc, "__dict__") else {}
            name = d.get("tool_name", "unknown")
            args = d.get("arguments", {})
            if not isinstance(args, dict):
                try:
                    args = json.loads(args) if isinstance(args, str) else {}
                except Exception:
                    args = {} if args is None else {"raw": str(args)}
            out.append({
                "tool_name": name,
                "arguments": args,
                "response": serialize_for_json(d.get("response")),
            })
    return out
