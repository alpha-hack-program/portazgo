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
portazgo: Pluggable agent backends for RAGAS dataset generation.

Use Agent(type="default") for Llama Stack Responses API, or Agent(type="lang-graph")
for a future LangGraph-based implementation.
"""

from portazgo.agent import Agent, AgentType
from portazgo.chats import ChatMessage, format_history_as_prefix
from portazgo.llama_utils import discover_mcp_tools, list_vector_store_names, resolve_vector_store_id
from portazgo.utils import extract_tool_calls, serialize_for_json, strip_think_blocks

__all__ = [
    "Agent",
    "AgentType",
    "ChatMessage",
    "discover_mcp_tools",
    "extract_tool_calls",
    "list_vector_store_names",
    "format_history_as_prefix",
    "resolve_vector_store_id",
    "serialize_for_json",
    "strip_think_blocks",
]
__version__ = "0.1.0"
