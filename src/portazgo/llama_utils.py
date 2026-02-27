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

"""
Llama Stack client utilities: MCP tool discovery, vector store resolution.
"""

from llama_stack_client.types.vector_store import VectorStore


from datetime import datetime
from typing import Any, Dict, List

from llama_stack_client import LlamaStackClient


def list_vector_store_names(client: LlamaStackClient) -> List[str]:
    """Return unique names of vector stores from Llama Stack, sorted alphabetically."""
    all_vector_stores = list(client.vector_stores.list())
    names = set()
    for vs in all_vector_stores:
        name = getattr(vs, "name", None)
        if name is not None:
            names.add(name)
    return sorted(names)


def _created_at_key(vs: Any) -> Any:
    """Extract creation timestamp for sorting; fallback to datetime.min if missing."""
    val = getattr(vs, "created_at", None) or getattr(vs, "createdAt", None)
    return val if val is not None else datetime.min


def discover_mcp_tools(client: LlamaStackClient, tools: str) -> List[Dict[str, Any]]:
    """Discover MCP tools from Llama Stack. tools: '' or 'none' (no MCP tools), 'all', or 'tool1,tool2'."""
    tool_filter = (tools or "").strip().lower()
    if not tool_filter or tool_filter == "none":
        return []
    tool_groups = list(client.toolgroups.list())
    requested = [] if tool_filter == "all" else [t.strip().lower() for t in tools.split(",") if t.strip()]
    mcp_tools = []
    for group in tool_groups:
        if not getattr(group, "identifier", "").startswith("mcp::"):
            continue
        if getattr(group, "provider_id", None) and getattr(group, "provider_id") != "model-context-protocol":
            continue
        tool_name = (group.identifier.split("::", 1)[1] if "::" in group.identifier else group.identifier).lower()
        if requested and tool_name not in requested:
            continue
        mcp_endpoint = getattr(group, "mcp_endpoint", None)
        server_url = None
        if mcp_endpoint:
            server_url = getattr(mcp_endpoint, "uri", None) or (mcp_endpoint.get("uri") if isinstance(mcp_endpoint, dict) else None)
        if server_url:
            mcp_tools.append({
                "type": "mcp",
                "server_label": tool_name,
                "server_url": server_url,
            })
    return mcp_tools


def find_latest_vector_store_id_by_name(
    client: LlamaStackClient,
    vector_store_name: str | None = None,
) -> str:
    """
    Resolve vector store to ID via Llama Stack.
    Same behavior as the search command: if name is given, use latest store with that name;
    if None, use latest store overall (any name).
    """
    all_vector_stores = list[VectorStore](client.vector_stores.list())
    if not all_vector_stores:
        raise ValueError(
            "No vector stores found. Create one first (e.g. via the load command)."
        )
    if vector_store_name:
        matching = [
            vs for vs in all_vector_stores
            if hasattr(vs, "name") and vs.name == vector_store_name
        ]
        if not matching:
            available = [getattr(vs, "name", None) for vs in all_vector_stores if hasattr(vs, "name")]
            raise ValueError(
                f"No vector store found with name '{vector_store_name}'. "
                f"Available: {available}"
            )
        stores = matching
    else:
        stores = all_vector_stores
    try:
        stores.sort(key=_created_at_key, reverse=True)
    except (TypeError, ValueError):
        pass
    chosen = stores[0]
    return chosen.id


# Alias for backward compatibility / __init__ exports
resolve_vector_store_id = find_latest_vector_store_id_by_name
