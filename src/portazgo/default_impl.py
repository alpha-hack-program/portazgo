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
Default agent implementation using Llama Stack Responses API.

Same logic as generate_ragas_dataset in ragas_pipeline.py and ragas_dataset_generator.py.
"""

import json
import logging
import os
from typing import Any, Dict, Iterator, List

from portazgo.chats import format_history_as_prefix
from portazgo.utils import extract_tool_calls, strip_think_blocks

logger = logging.getLogger(__name__)
_log_level = getattr(logging, (os.environ.get("LOG_LEVEL") or "INFO").strip().upper(), logging.INFO)
logger.setLevel(_log_level)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setLevel(_log_level)
    _h.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(_h)
    logger.propagate = False  # avoid duplicate lines when root is also configured


def _build_tools_list(
    vector_store_id: str,
    mcp_tools: List[Dict[str, Any]],
    *,
    ranker: str = "default",
    retrieval_mode: str = "vector",
    file_search_max_chunks: int = 5,
    file_search_score_threshold: float = 0.7,
    file_search_max_tokens_per_chunk: int = 512,
    include_file_search: bool = True,
) -> List[Dict[str, Any]]:
    """Build the tools list for Responses API (shared by invoke and generate_ragas_dataset)."""
    tools_list: List[Dict[str, Any]] = []
    if vector_store_id and include_file_search:
        tools_list.append({
            "type": "file_search",
            "vector_store_ids": [vector_store_id],
            "filters": {},
            "max_num_results": file_search_max_chunks,
            "max_chunks": file_search_max_chunks,
            "retrieval_mode": retrieval_mode,
            "max_tokens_per_chunk": file_search_max_tokens_per_chunk,
            "ranking_options": {
                "ranker": ranker,
                "score_threshold": file_search_score_threshold,
            },
        })
    tools_list.extend(mcp_tools)
    return tools_list


def _effective_input(input_text: str, messages: List[Dict[str, str]] | None) -> str:
    """Build the effective input: optional conversation prefix + current user input."""
    if not messages:
        return input_text
    prefix = format_history_as_prefix(messages)
    return prefix + "User: " + input_text.strip()


def _format_context_and_query(chunks: List[str], query: str) -> str:
    """Format retrieved RAG chunks and the user query as structured input for the LLM."""
    if not chunks:
        return query
    context_block = "\n\n---\n\n".join(chunks)
    return f"""Context (retrieved from the knowledge base for the following query):

{context_block}

---

Query: {query.strip()}"""


def get_rag_context(
    client: Any,
    vector_store_id: str,
    query: str,
    *,
    max_results: int = 10,
    score_threshold: float = 0.7,
    ranker: str = "default",
    search_mode: str = "vector",
) -> List[str]:
    """
    Retrieve RAG chunks from a vector store to prepare a subsequent call to the Responses API.

    Uses the vector store search API and returns a list of chunk texts in order of relevance.
    Callers can pass these chunks into instructions, concatenate them for input, or use
    them when building the tools/request for client.responses.create().

    Args:
        client: The Llama Stack client (must provide vector_stores.search).
        vector_store_id: ID of the vector store to search.
        query: Search query.
        max_results: Maximum number of chunks to return (default 10).
        score_threshold: Minimum score for results (default 0.7).
        ranker: Ranker name (default "default").
        search_mode: Search mode, e.g. "vector" or "hybrid" (default "vector").

    Returns:
        List of chunk text strings, one per retrieved content item. Empty list if no results
        or if vector_store_id is empty.
    """
    if not vector_store_id:
        logger.warning("No vector store ID provided for query: %s", query)
        return []
    from llama_stack_client.types.vector_store_search_params import RankingOptions

    ranking_options = RankingOptions(ranker=ranker, score_threshold=score_threshold)
    search_response = client.vector_stores.search(
        vector_store_id=vector_store_id,
        query=query,
        search_mode=search_mode,
        ranking_options=ranking_options,
        max_num_results=max_results,
    )
    chunks: List[str] = []
    for data in getattr(search_response, "data", []) or []:
        for content in getattr(data, "content", []) or []:
            text = getattr(content, "text", None)
            if text and isinstance(text, str):
                chunks.append(text)
    logger.debug("Retrieved %d chunks for query: %s", len(chunks), query)
    logger.debug("Chunks: %s", [c[:50] + "..." if len(c) > 50 else c for c in chunks])
    return chunks


def invoke(
    input_text: str,
    client: Any,
    model_id: str,
    vector_store_id: str,
    mcp_tools: List[Dict[str, Any]],
    *,
    messages: List[Dict[str, str]] | None = None,
    instructions: str,
    force_file_search: bool = False,
    ranker: str = "default",
    retrieval_mode: str = "vector",
    file_search_max_chunks: int = 5,
    file_search_score_threshold: float = 0.7,
    file_search_max_tokens_per_chunk: int = 512,
    strip_think_blocks: bool = True,
) -> Dict[str, Any]:
    """
    Resolve a single input with the given tools and vector store (one request).

    Returns a single result dict: answer (str), contexts (list[str]), tool_calls (list[dict]).
    """

    # If force_file_search is True, retrieve chunks and inject as structured context; do not add file_search tool
    if force_file_search:
        chunks = get_rag_context(
            client=client,
            vector_store_id=vector_store_id,
            query=input_text,
            max_results=file_search_max_chunks,
            score_threshold=file_search_score_threshold,
            ranker=ranker,
        )
        input_text = _format_context_and_query(chunks, input_text)

    effective = _effective_input(input_text, messages)
    tools_list = _build_tools_list(
        vector_store_id,
        mcp_tools,
        ranker=ranker,
        retrieval_mode=retrieval_mode,
        file_search_max_chunks=file_search_max_chunks,
        file_search_score_threshold=file_search_score_threshold,
        file_search_max_tokens_per_chunk=file_search_max_tokens_per_chunk,
        include_file_search=not force_file_search,
    )
    request_config: Dict[str, Any] = {
        "model": model_id,
        "input": effective,
        "tools": tools_list,
        # "tool_choice": {
        #     "type": "function",
        #     "function": {
        #         "name": "file_search"  # must match your tool's defined name exactly
        #     }
        # }
    }
    if instructions and instructions.strip():
        request_config["instructions"] = instructions.strip()
    if any(t.get("type") == "file_search" for t in tools_list):
        request_config["include"] = ["file_search_call.results"]

    response = client.responses.create(**request_config)
    answer = getattr(response, "output_text", str(response))
    if isinstance(answer, str) and strip_think_blocks:
        answer = strip_think_blocks(answer)

    contexts: List[str] = []
    if hasattr(response, "output") and isinstance(response.output, list):
        for output_item in response.output:
            if hasattr(output_item, "results") and isinstance(output_item.results, list):
                for result in output_item.results:
                    if hasattr(result, "text") and result.text:
                        contexts.append(result.text)

    tool_calls = extract_tool_calls(response)
    result: Dict[str, Any] = {
        "answer": answer,
        "contexts": contexts if contexts else [],
        "tool_calls": tool_calls,
    }
    for tc in tool_calls:
        if tc.get("tool_name") != "file_search":
            resp = tc.get("response")
            if resp is not None:
                ctx = resp if isinstance(resp, str) else json.dumps(resp, ensure_ascii=False)
                result["contexts"].append(ctx)
    return result


def invoke_stream(
    input_text: str,
    client: Any,
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
    strip_think_blocks: bool = True,
) -> Iterator[Dict[str, Any]]:
    """
    Same as invoke but yields stream events: content_delta then done.
    If the backend does not support token-level streaming, yields one delta then done.
    """
    if force_file_search:
        chunks = get_rag_context(
            client=client,
            vector_store_id=vector_store_id,
            query=input_text,
            max_results=file_search_max_chunks,
            score_threshold=file_search_score_threshold,
            ranker=ranker,
        )
        input_text = _format_context_and_query(chunks, input_text)
    effective = _effective_input(input_text, messages)
    tools_list = _build_tools_list(
        vector_store_id,
        mcp_tools,
        ranker=ranker,
        retrieval_mode=retrieval_mode,
        file_search_max_chunks=file_search_max_chunks,
        file_search_score_threshold=file_search_score_threshold,
        file_search_max_tokens_per_chunk=file_search_max_tokens_per_chunk,
        include_file_search=not force_file_search,
    )
    request_config: Dict[str, Any] = {
        "model": model_id,
        "input": effective,
        "tools": tools_list,
    }
    if instructions and instructions.strip():
        request_config["instructions"] = instructions.strip()
    if any(t.get("type") == "file_search" for t in tools_list):
        request_config["include"] = ["file_search_call.results"]

    stream_supported = False
    try:
        resp_stream = client.responses.create(**request_config, stream=True)
        if hasattr(resp_stream, "__iter__") and not isinstance(resp_stream, (str, bytes)):
            stream_supported = True
    except TypeError:
        pass

    if stream_supported:
        buffer: List[str] = []
        last_response: Any = None
        for chunk in resp_stream:
            chunk_type = getattr(chunk, "type", None)
            if chunk_type == "response.completed":
                last_response = getattr(chunk, "response", None)
                continue
            text = getattr(chunk, "output_text", None) or getattr(chunk, "text", None) or getattr(chunk, "delta", "")
            if text and isinstance(text, str):
                # Use raw text for streaming: strip_think_blocks(text).strip() would remove
                # leading/trailing spaces from chunks like " tax", causing "Thetaxrate" instead of "The tax rate"
                buffer.append(text)
                yield {"type": "content_delta", "delta": text}
        full_answer = "".join(buffer)
        if strip_think_blocks:
            full_answer = strip_think_blocks(full_answer)
        tool_calls: List[Dict[str, Any]] = []
        contexts: List[str] = []
        if last_response:
            tool_calls = extract_tool_calls(last_response)
            if hasattr(last_response, "output") and isinstance(last_response.output, list):
                for output_item in last_response.output:
                    if hasattr(output_item, "results") and isinstance(output_item.results, list):
                        for result in output_item.results:
                            if hasattr(result, "text") and result.text:
                                contexts.append(result.text)
            for tc in tool_calls:
                if tc.get("tool_name") != "file_search":
                    resp = tc.get("response")
                    if resp is not None:
                        ctx = resp if isinstance(resp, str) else json.dumps(resp, ensure_ascii=False)
                        contexts.append(ctx)
        yield {"type": "done", "answer": full_answer, "contexts": contexts, "tool_calls": tool_calls}
        return

    # input_text already has context injected when force_file_search; pass False to avoid double work
    result = invoke(
        input_text=input_text,
        client=client,
        model_id=model_id,
        vector_store_id=vector_store_id,
        mcp_tools=mcp_tools,
        messages=messages,
        instructions=instructions,
        force_file_search=False,
        ranker=ranker,
        retrieval_mode=retrieval_mode,
        file_search_max_chunks=file_search_max_chunks,
        file_search_score_threshold=file_search_score_threshold,
        file_search_max_tokens_per_chunk=file_search_max_tokens_per_chunk,
        strip_think_blocks=strip_think_blocks,
    )
    answer = result["answer"]
    if answer:
        yield {"type": "content_delta", "delta": answer}
    yield {"type": "done", "answer": answer, "contexts": result["contexts"], "tool_calls": result["tool_calls"]}


def generate_ragas_dataset(
    base_dataset: List[Dict[str, Any]],
    client: Any,
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
    Generate RAGAS dataset by querying Llama Stack Responses API for each question.

    Returns list of RAGAS entries: id, question, answer, contexts, ground_truth, optional tool_calls, etc.
    """
    tools_list = _build_tools_list(
        vector_store_id,
        mcp_tools,
        ranker=ranker,
        retrieval_mode=retrieval_mode,
        file_search_max_chunks=file_search_max_chunks,
        file_search_score_threshold=file_search_score_threshold,
        file_search_max_tokens_per_chunk=file_search_max_tokens_per_chunk,
        include_file_search=not force_file_search,
    )

    ragas_dataset: List[Dict[str, Any]] = []
    errors: List[str] = []
    error_count = 0

    for i, item in enumerate(base_dataset, 1):
        question_id = item.get("id", f"q_{i}")
        question = item["question"]
        ground_truth = item.get("ground_truth", "")

        try:
            input_for_request = question
            prefetched_chunks: List[str] = []
            if force_file_search:
                prefetched_chunks = get_rag_context(
                    client=client,
                    vector_store_id=vector_store_id,
                    query=question,
                    max_results=file_search_max_chunks,
                    score_threshold=file_search_score_threshold,
                    ranker=ranker,
                )
                input_for_request = _format_context_and_query(prefetched_chunks, question)
            request_config: Dict[str, Any] = {
                "model": model_id,
                "input": input_for_request,
                "tools": tools_list,
            }
            if instructions and instructions.strip():
                request_config["instructions"] = instructions.strip()
            if any(t.get("type") == "file_search" for t in tools_list):
                request_config["include"] = ["file_search_call.results"]

            response = client.responses.create(**request_config)
            answer = getattr(response, "output_text", str(response))
            if isinstance(answer, str):
                answer = strip_think_blocks(answer)

            contexts: List[str] = []
            if force_file_search and prefetched_chunks:
                contexts = list(prefetched_chunks)
            elif hasattr(response, "output") and isinstance(response.output, list):
                for output_item in response.output:
                    if hasattr(output_item, "results") and isinstance(output_item.results, list):
                        for result in output_item.results:
                            if hasattr(result, "text") and result.text:
                                contexts.append(result.text)

            tool_calls = extract_tool_calls(response)
            ragas_entry: Dict[str, Any] = {
                "id": question_id,
                "question": question,
                "answer": answer,
                "contexts": contexts if contexts else [],
                "ground_truth": ground_truth,
            }
            if tool_calls:
                ragas_entry["tool_calls"] = tool_calls
                for tc in tool_calls:
                    if tc.get("tool_name") != "file_search":
                        resp = tc.get("response")
                        if resp is not None:
                            ctx = resp if isinstance(resp, str) else json.dumps(resp, ensure_ascii=False)
                            ragas_entry["contexts"].append(ctx)
            if "difficulty" in item:
                ragas_entry["difficulty"] = item["difficulty"]
            if item.get("expected_tool"):
                ragas_entry["expected_tool"] = item["expected_tool"]
            if item.get("expected_tool_parameters"):
                ragas_entry["expected_tool_parameters"] = item["expected_tool_parameters"]

            ragas_dataset.append(ragas_entry)

        except Exception as e:
            error_count += 1
            errors.append(f"{question_id}: {str(e)}")
            ragas_dataset.append({
                "id": question_id,
                "question": question,
                "answer": f"ERROR: {str(e)}",
                "contexts": [],
                "ground_truth": ground_truth,
                "difficulty": item.get("difficulty", "unknown"),
                "error": True,
            })

    total = len(base_dataset)
    if error_count == total:
        raise ValueError(
            f"All {total} questions failed. First error: {errors[0] if errors else 'Unknown'}"
        )
    if error_count > total / 2:
        raise ValueError(
            f"Too many failures: {error_count}/{total}. Errors: {errors[:5]}"
        )

    return ragas_dataset
