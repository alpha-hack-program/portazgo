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
Agent abstraction with pluggable backends (default vs lang-graph).
"""

from typing import Any, Dict, Iterator, List, Literal

from portazgo import default_impl, lang_graph

AgentType = Literal["default", "lang-graph"]

_BACKENDS: Dict[AgentType, Any] = {
    "default": default_impl,
    "lang-graph": lang_graph,
}


class Agent:
    """
    Agent for RAGAS dataset generation with selectable backend.

    Parameters
    ----------
    type : "default" | "lang-graph"
        - "default": Uses Llama Stack Responses API (same logic as
          ragas_pipeline.generate_ragas_dataset and ragas_dataset_generator).
        - "lang-graph": LangGraph-based implementation (stub; raises
          NotImplementedError until implemented).
    """

    def __init__(self, type: AgentType = "default") -> None:
        if type not in _BACKENDS:
            raise ValueError(
                f"Agent type must be one of {list(_BACKENDS.keys())}, got {type!r}"
            )
        self._type = type
        self._backend = _BACKENDS[type]

    @property
    def type(self) -> AgentType:
        """Current agent backend type."""
        return self._type

    def generate_ragas_dataset(
        self,
        client: Any,
        base_dataset: List[Dict[str, Any]],
        model_id: str,
        vector_store_id: str,
        mcp_tools: List[Dict[str, Any]],
        *,
        instructions: str = "",
        ranker: str = "default",
        retrieval_mode: str = "vector",
        force_file_search: bool = False,
        file_search_max_chunks: int = 5,
        file_search_score_threshold: float = 0.7,
        file_search_max_tokens_per_chunk: int = 512,
    ) -> List[Dict[str, Any]]:
        """
        Generate a RAGAS-compatible dataset by querying the RAG system.

        Args:
            base_dataset: List of items with "question"; optional "id", "ground_truth".
            client: LlamaStackClient (for default) or backend-specific client.
            model_id: Model identifier for inference.
            vector_store_id: Vector store ID for file_search.
            mcp_tools: List of MCP tool config dicts.
            instructions: Optional system prompt.
            force_file_search: If True, pre-fetch RAG chunks and inject as context (no file_search tool).
            ranker: Ranker for file_search (default backend).
            retrieval_mode: "vector", "text", or "hybrid".
            file_search_max_chunks: Max chunks to retrieve.
            file_search_score_threshold: Min score for results (0–1).
            file_search_max_tokens_per_chunk: Max tokens per chunk.

        Returns:
            List of RAGAS entries: id, question, answer, contexts, ground_truth, etc.
        """
        return self._backend.generate_ragas_dataset(
            base_dataset=base_dataset,
            client=client,
            model_id=model_id,
            vector_store_id=vector_store_id,
            mcp_tools=mcp_tools,
            instructions=instructions,
            ranker=ranker,
            force_file_search=force_file_search,
            retrieval_mode=retrieval_mode,
            file_search_max_chunks=file_search_max_chunks,
            file_search_score_threshold=file_search_score_threshold,
            file_search_max_tokens_per_chunk=file_search_max_tokens_per_chunk,
        )

    def invoke(
        self,
        client: Any,
        input_text: str,
        model_id: str,
        vector_store_id: str,
        mcp_tools: List[Dict[str, Any]] | None = None,
        *,
        messages: List[Dict[str, str]] | None = None,
        instructions: str = "",
        force_file_search: bool = False,
        ranker: str = "default",
        retrieval_mode: str = "vector",
        file_search_max_chunks: int = 5,
        file_search_score_threshold: float = 0.7,
        file_search_max_tokens_per_chunk: int = 512,
    ) -> Dict[str, Any]:
        """
        Resolve a single input with the given tools and vector store (normal agent call).

        Similar shape to generate_ragas_dataset but for one query. Named after
        LangChain/LangGraph's invoke: run the agent once and return one result.

        Args:
            input_text: User question or message.
            client: LlamaStackClient (for default) or backend-specific client.
            model_id: Model identifier for inference.
            vector_store_id: Vector store ID for file_search.
            mcp_tools: List of MCP tool config dicts (default: empty list).
            messages: Optional conversation history for chatbots (list of {"role", "content"}).
            instructions: Optional system prompt.
            force_file_search: If True, pre-fetch RAG chunks and inject as context (no file_search tool).
            ranker: Ranker for file_search (default backend).
            retrieval_mode: "vector", "text", or "hybrid".
            file_search_max_chunks: Max chunks to retrieve.
            file_search_score_threshold: Min score for results (0–1).
            file_search_max_tokens_per_chunk: Max tokens per chunk.

        Returns:
            Dict with keys: answer (str), contexts (list[str]), tool_calls (list[dict]).
        """
        if mcp_tools is None:
            mcp_tools = []
        return self._backend.invoke(
            input_text=input_text,
            client=client,
            model_id=model_id,
            vector_store_id=vector_store_id,
            mcp_tools=mcp_tools,
            messages=messages,
            instructions=instructions,
            force_file_search=force_file_search,
            ranker=ranker,
            retrieval_mode=retrieval_mode,
            file_search_max_chunks=file_search_max_chunks,
            file_search_score_threshold=file_search_score_threshold,
            file_search_max_tokens_per_chunk=file_search_max_tokens_per_chunk,
        )

    def invoke_stream(
        self,
        client: Any,
        input_text: str,
        model_id: str,
        vector_store_id: str,
        mcp_tools: List[Dict[str, Any]] | None = None,
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
        """
        Same as invoke but yields stream events for real-time display (e.g. Streamlit).

        Yields: {"type": "content_delta", "delta": str} for each chunk;
        then {"type": "done", "answer": str, "contexts": list, "tool_calls": list}.
        """
        if mcp_tools is None:
            mcp_tools = []
        return self._backend.invoke_stream(
            input_text=input_text,
            client=client,
            model_id=model_id,
            vector_store_id=vector_store_id,
            mcp_tools=mcp_tools,
            messages=messages,
            instructions=instructions,
            force_file_search=force_file_search,
            ranker=ranker,
            retrieval_mode=retrieval_mode,
            file_search_max_chunks=file_search_max_chunks,
            file_search_score_threshold=file_search_score_threshold,
            file_search_max_tokens_per_chunk=file_search_max_tokens_per_chunk,
        )
