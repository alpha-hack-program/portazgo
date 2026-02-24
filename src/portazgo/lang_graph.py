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
LangGraph-based agent implementation (stub).

Use agent type "lang-graph" to select this backend. Implement this module
to generate RAGAS datasets via a LangGraph flow instead of Llama Stack Responses API.
"""

from typing import Any, Dict, Iterator, List


def invoke(
    client: Any,
    input_text: str,
    model_id: str,
    vector_store_id: str,
    mcp_tools: List[Dict[str, Any]],
    *,
    messages: List[Dict[str, str]] | None = None,
    instructions: str = "",
    ranker: str = "default",
    retrieval_mode: str = "vector",
    file_search_max_chunks: int = 5,
    file_search_score_threshold: float = 0.7,
    file_search_max_tokens_per_chunk: int = 512,
) -> Dict[str, Any]:
    """
    Resolve a single input using a LangGraph agent (placeholder).

    Raises NotImplementedError. Implement this when integrating LangGraph.
    """
    raise NotImplementedError(
        "LangGraph agent backend is not yet implemented. "
        "Use type='default' or implement portazgo.lang_graph.invoke."
    )


def invoke_stream(
    client: Any,
    input_text: str,
    model_id: str,
    vector_store_id: str,
    mcp_tools: List[Dict[str, Any]],
    *,
    messages: List[Dict[str, str]] | None = None,
    instructions: str = "",
    force_file_search: bool = False,
    ranker: str = "default",
    retrieval_mode: str = "vector",
    file_search_max_chunks: int = 5,
    file_search_score_threshold: float = 0.7,
    file_search_max_tokens_per_chunk: int = 512,
) -> Iterator[Dict[str, Any]]:
    """Streaming invoke for LangGraph (placeholder). Raises NotImplementedError."""
    raise NotImplementedError(
        "LangGraph agent backend is not yet implemented. "
        "Use type='default' or implement portazgo.lang_graph.invoke_stream."
    )


def generate_ragas_dataset(
    client: Any,
    base_dataset: List[Dict[str, Any]],
    model_id: str,
    vector_store_id: str,
    mcp_tools: List[Dict[str, Any]],
    *,
    instructions: str = "",
    ranker: str = "default",
    retrieval_mode: str = "vector",
    file_search_max_chunks: int = 5,
    file_search_score_threshold: float = 0.7,
    file_search_max_tokens_per_chunk: int = 512,
    force_file_search: bool = False,
) -> List[Dict[str, Any]]:
    """
    Generate RAGAS dataset using a LangGraph agent (placeholder).

    Raises NotImplementedError. Implement this when integrating LangGraph
    as an alternative to the default Llama Stack Responses API path.
    """
    raise NotImplementedError(
        "LangGraph agent backend is not yet implemented. "
        "Use type='default' or implement portazgo.lang_graph.generate_ragas_dataset."
    )
