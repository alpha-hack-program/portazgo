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

Supports patterns: simple (single call) and plan_execute (planner selects tools → executor).
"""

import json
import logging
import os
import re
from typing import Any, Callable, Dict, Iterator, List, Literal

from portazgo.chats import format_history_as_prefix
from portazgo.utils import extract_tool_calls, strip_think_blocks as _strip_think_blocks
from portazgo.validation import default_validator

PatternType = Literal["simple", "plan_execute"]

logger = logging.getLogger(__name__)

_TOOL_SELECTOR_INSTRUCTIONS = """You are a tool selector. Based on the user question and the retrieved context (from file_search), decide which MCP toolgroups (if any) are needed.

IMPORTANT: Do NOT call any tools. Output ONLY a valid JSON object. No function calls.

When to select tools:
- COMPUTATION needed (e.g. "calculate tax for 90000 income", "what is the total tax liability for X") → select tax-engine.
- LOOKUP only (e.g. "what is the tax rate", "what does the law say") when the answer is in the retrieved context → select [].

Examples:
- "If a taxpayer earns 90000, what is their total tax liability?" → {"tools": ["tax-engine"]}
- "What is the tax rate for the first bracket?" → {"tools": []}

Categorize the question into one of the following categories:
- COMPUTATION: A computation is needed to answer the question.
- LOOKUP: A lookup is needed to answer the question.
- OTHER: The question does not fit into the above categories.

If the question is a COMPUTATION, select the toolgroup that is needed to answer the question.
If the question is a LOOKUP, select the toolgroup that is needed to answer the question.
If the question is OTHER, select [].

Output a JSON object with a single key "tools" whose value is an array of toolgroup names.
Use ONLY toolgroup names from the list below. If no tools are needed, output {"tools": []}."""

_FORCE_TOOL_USE_INSTRUCTION = """IMPORTANT: You MUST call the following tool(s) to answer the question. Do not attempt to answer without invoking them first. The planner has determined these tools are required for this task."""


def _build_force_tool_instruction(tool_names: List[str]) -> str:
    """Build instruction to force the model to call the given tools (used when planner selected them)."""
    if not tool_names:
        return ""
    names = ", ".join(tool_names)
    return f"{_FORCE_TOOL_USE_INSTRUCTION}\nRequired tools: {names}"


def _select_tools(
    client: Any,
    model_id: str,
    vector_store_id: str,
    input_text: str,
    mcp_tools: List[Dict[str, Any]],
    *,
    ranker: str = "default",
    retrieval_mode: str = "vector",
    file_search_max_chunks: int = 5,
    file_search_score_threshold: float = 0.7,
    file_search_max_tokens_per_chunk: int = 512,
) -> List[Dict[str, Any]]:
    """
    Planner phase: call Responses API with file_search only; parse response for tool selection.
    Returns the subset of mcp_tools that were selected.
    """
    if not mcp_tools:
        return []

    from portazgo.llama_utils import get_mcp_tool_schemas

    tool_names = [t.get("server_label", "").lower() for t in mcp_tools if t.get("server_label")]
    if not tool_names:
        return []

    schemas = get_mcp_tool_schemas(client, mcp_tools)
    tools_block_parts = ["Available MCP toolgroups (select by name; do not call any tools):"]
    ordered_names = sorted(tool_names, key=lambda x: (0 if x == "finance-engine" else 1, x))
    for server_label in ordered_names:
        tools = schemas.get(server_label, [{"name": server_label, "description": server_label}])
        descs = [t.get("description", "") for t in tools if t.get("description")]
        agg = "; ".join(descs) if descs else server_label
        tools_block_parts.append(f"- {server_label}: {agg}")
    tools_block = "\n".join(tools_block_parts)
    selector_instructions = _TOOL_SELECTOR_INSTRUCTIONS + f"\n\n{tools_block}"

    tools_list = _build_tools_list(
        vector_store_id,
        [],
        ranker=ranker,
        retrieval_mode=retrieval_mode,
        file_search_max_chunks=file_search_max_chunks,
        file_search_score_threshold=file_search_score_threshold,
        file_search_max_tokens_per_chunk=file_search_max_tokens_per_chunk,
        include_file_search=True,
    )

    request_config: Dict[str, Any] = {
        "model": model_id,
        "input": input_text.strip(),
        "tools": tools_list,
        "instructions": selector_instructions.strip(),
    }
    request_config["include"] = ["file_search_call.results"]

    try:
        logger.info("plan_execute: planner calling Responses API with config: %s", request_config)
        response = client.responses.create(**request_config)
        logger.info("plan_execute: planner call successful")
        raw = getattr(response, "output_text", str(response))
        if not isinstance(raw, str):
            raw = str(raw)
        logger.info("plan_execute: planner response: %s", raw)
    except Exception as e:
        logger.warning("plan_execute: planner call failed (%s), falling back to all tools", e)
        return list(mcp_tools)

    raw = getattr(response, "output_text", str(response))
    if not isinstance(raw, str):
        raw = str(raw)
    raw = _strip_think_blocks(raw)

    selected_names: List[str] = []
    match = re.search(r'\{[^{}]*"tools"\s*:\s*\[[^\]]*\]\s*\}', raw, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            parsed = json.loads(match.group())
            tools_arr = parsed.get("tools", [])
            if isinstance(tools_arr, list):
                selected_names = [str(t).strip().lower() for t in tools_arr if t]
        except json.JSONDecodeError:
            pass

    name_to_tool = {t.get("server_label", "").lower(): t for t in mcp_tools if t.get("server_label")}
    tool_to_server: Dict[str, str] = {}
    for label, tools in schemas.items():
        for t in tools:
            tool_to_server[(t.get("name") or "").lower()] = label
        tool_to_server[label] = label
    server_labels = set(name_to_tool.keys())
    selected = []
    for n in selected_names:
        server = tool_to_server.get(n) or (n if n in server_labels else None)
        if server and server in name_to_tool and name_to_tool[server] not in selected:
            selected.append(name_to_tool[server])
    logger.info("plan_execute: planner selected tools %s from %s", [t.get("server_label") for t in selected], tool_names)
    return selected


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
    """
    Build the tools list for Responses API (shared by invoke and generate_ragas_dataset).
    It includes the file_search tool if include_file_search is True and vector_store_id is not empty.
    It includes the MCP tools if mcp_tools is not empty.

    Args:
        vector_store_id: ID of the vector store to search.
        mcp_tools: List of MCP tools to include.
        ranker: Ranker name (default "default").
        retrieval_mode: Search mode, e.g. "vector" or "hybrid" (default "vector").
        file_search_max_chunks: Maximum number of chunks to return (default 5).
        file_search_score_threshold: Minimum score for results (default 0.7).
        file_search_max_tokens_per_chunk: Maximum number of tokens per chunk (default 512).
        include_file_search: Whether to include the file_search tool (default True).

    Returns:
        List of tools to include in the request to the Responses API.
    """
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
    instructions: str = "",
    force_file_search: bool = False,
    ranker: str = "default",
    retrieval_mode: str = "vector",
    file_search_max_chunks: int = 5,
    file_search_score_threshold: float = 0.7,
    file_search_max_tokens_per_chunk: int = 512,
    strip_think_blocks: bool = True,
    pattern: PatternType = "simple",
    force_tool_use: bool = False,
    validate_output: bool = False,
    max_validation_retries: int = 1,
    validation_rules: Callable[[str, str], tuple[bool, str]] | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Resolve a single input with the given tools and vector store.

    pattern: "simple" (single call) or "plan_execute" (planner selects tools, then executor).
    Returns a single result dict: answer (str), contexts (list[str]), tool_calls (list[dict]).
    When validate_output=True, retries up to max_validation_retries with feedback on failure.
    """
    if pattern == "plan_execute" and mcp_tools:
        effective_input = _effective_input(input_text.strip(), messages)
        selected_tools = _select_tools(
            client=client,
            model_id=model_id,
            vector_store_id=vector_store_id,
            input_text=effective_input,
            mcp_tools=mcp_tools,
            ranker=ranker,
            retrieval_mode=retrieval_mode,
            file_search_max_chunks=file_search_max_chunks,
            file_search_score_threshold=file_search_score_threshold,
            file_search_max_tokens_per_chunk=file_search_max_tokens_per_chunk,
        )
        mcp_tools = selected_tools
        force_tool_use = force_tool_use or bool(selected_tools)

    validator = validation_rules if validation_rules is not None else default_validator
    original_query = input_text.strip()

    # If force_file_search is True, retrieve chunks and inject as structured context; do not add file_search tool
    prefetched_chunks: List[str] = []
    if force_file_search:
        prefetched_chunks = get_rag_context(
            client=client,
            vector_store_id=vector_store_id,
            query=original_query,
            max_results=file_search_max_chunks,
            score_threshold=file_search_score_threshold,
            ranker=ranker,
        )

    current_query = original_query
    last_result: Dict[str, Any] | None = None

    for attempt in range(max_validation_retries + 1):
        if force_file_search and prefetched_chunks:
            effective_input = _format_context_and_query(prefetched_chunks, current_query)
        else:
            effective_input = current_query
        effective = _effective_input(effective_input, messages)

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
        if force_tool_use and mcp_tools:
            tool_names = [t.get("server_label", "") for t in mcp_tools if t.get("server_label")]
            if tool_names:
                force_instr = _build_force_tool_instruction(tool_names)
                existing = request_config.get("instructions", "")
                request_config["instructions"] = (existing + "\n\n" + force_instr).strip()
        if any(t.get("type") == "file_search" for t in tools_list):
            request_config["include"] = ["file_search_call.results"]

        response = client.responses.create(**request_config)
        answer = getattr(response, "output_text", str(response))
        if isinstance(answer, str) and strip_think_blocks:
            answer = _strip_think_blocks(answer)

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

        last_result = result

        if not validate_output:
            return result

        passed, feedback = validator(result["answer"], original_query)
        if passed:
            return result

        if attempt >= max_validation_retries:
            return result

        current_query = f"{original_query}\n\n[Validation feedback - please improve your answer]: {feedback}"

    return last_result or {}


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
    pattern: PatternType = "simple",
    validate_output: bool = False,
    max_validation_retries: int = 1,
    validation_rules: Any = None,
    **kwargs: Any,
) -> Iterator[Dict[str, Any]]:
    """
    Same as invoke but yields stream events: content_delta then done.
    plan_execute: planner (non-streaming) then executor (streaming).
    Validation is not supported: use invoke() with validate_output=True for non-streaming.
    """
    if validate_output:
        raise ValueError(
            "validate_output is not supported with invoke_stream. "
            "Use invoke() with validate_output=True for non-streaming calls with validation."
        )

    if pattern == "plan_execute" and mcp_tools:
        effective_input = _effective_input(input_text.strip(), messages)
        selected_tools = _select_tools(
            client=client,
            model_id=model_id,
            vector_store_id=vector_store_id,
            input_text=effective_input,
            mcp_tools=mcp_tools,
            ranker=ranker,
            retrieval_mode=retrieval_mode,
            file_search_max_chunks=file_search_max_chunks,
            file_search_score_threshold=file_search_score_threshold,
            file_search_max_tokens_per_chunk=file_search_max_tokens_per_chunk,
        )
        yield from _invoke_stream_impl(
            input_text=input_text,
            client=client,
            model_id=model_id,
            vector_store_id=vector_store_id,
            mcp_tools=selected_tools,
            messages=messages,
            instructions=instructions,
            force_file_search=force_file_search,
            ranker=ranker,
            retrieval_mode=retrieval_mode,
            file_search_max_chunks=file_search_max_chunks,
            file_search_score_threshold=file_search_score_threshold,
            file_search_max_tokens_per_chunk=file_search_max_tokens_per_chunk,
            strip_think_blocks=strip_think_blocks,
            force_tool_use=bool(selected_tools),
        )
        return

    yield from _invoke_stream_impl(
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
        strip_think_blocks=strip_think_blocks,
        force_tool_use=False,
    )


def _invoke_stream_impl(
    input_text: str,
    client: Any,
    model_id: str,
    vector_store_id: str,
    mcp_tools: List[Dict[str, Any]],
    *,
    messages: List[Dict[str, str]] | None = None,
    instructions: str = "",
    force_file_search: bool = False,
    force_tool_use: bool = False,
    ranker: str = "default",
    retrieval_mode: str = "vector",
    file_search_max_chunks: int = 5,
    file_search_score_threshold: float = 0.7,
    file_search_max_tokens_per_chunk: int = 512,
    strip_think_blocks: bool = True,
) -> Iterator[Dict[str, Any]]:
    """Internal streaming implementation (no pattern routing)."""
    prefetched_chunks: List[str] = []
    original_query = input_text.strip()
    if force_file_search:
        prefetched_chunks = get_rag_context(
            client=client,
            vector_store_id=vector_store_id,
            query=original_query,
            max_results=file_search_max_chunks,
            score_threshold=file_search_score_threshold,
            ranker=ranker,
        )
        input_text = _format_context_and_query(prefetched_chunks, original_query)
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
    if force_tool_use and mcp_tools:
        tool_names = [t.get("server_label", "") for t in mcp_tools if t.get("server_label")]
        if tool_names:
            force_instr = _build_force_tool_instruction(tool_names)
            existing = request_config.get("instructions", "")
            request_config["instructions"] = (existing + "\n\n" + force_instr).strip()
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
            full_answer = _strip_think_blocks(full_answer)
        tool_calls: List[Dict[str, Any]] = []
        contexts: List[str] = []
        if force_file_search and prefetched_chunks:
            contexts = list(prefetched_chunks)
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
    pattern: PatternType = "simple",
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Generate RAGAS dataset by querying Llama Stack Responses API for each question.

    pattern: "simple" or "plan_execute" (tool selection per question).
    Returns list of RAGAS entries: id, question, answer, contexts, ground_truth, optional tool_calls, etc.
    """
    if pattern != "plan_execute" or not mcp_tools:
        return _generate_ragas_dataset_simple(
            base_dataset=base_dataset,
            client=client,
            model_id=model_id,
            vector_store_id=vector_store_id,
            mcp_tools=mcp_tools,
            instructions=instructions,
            ranker=ranker,
            retrieval_mode=retrieval_mode,
            file_search_max_chunks=file_search_max_chunks,
            file_search_score_threshold=file_search_score_threshold,
            file_search_max_tokens_per_chunk=file_search_max_tokens_per_chunk,
            force_file_search=force_file_search,
        )

    # plan_execute: tool selection per question
    ragas_dataset: List[Dict[str, Any]] = []
    for item in base_dataset:
        question = item.get("question", "")
        question_id = item.get("id", f"q_{len(ragas_dataset) + 1}")
        ground_truth = item.get("ground_truth", "")
        try:
            effective_input = _effective_input(question, None)
            selected_tools = _select_tools(
                client=client,
                model_id=model_id,
                vector_store_id=vector_store_id,
                input_text=effective_input,
                mcp_tools=mcp_tools,
                ranker=ranker,
                retrieval_mode=retrieval_mode,
                file_search_max_chunks=file_search_max_chunks,
                file_search_score_threshold=file_search_score_threshold,
                file_search_max_tokens_per_chunk=file_search_max_tokens_per_chunk,
            )
            result = invoke(
                input_text=question,
                client=client,
                model_id=model_id,
                vector_store_id=vector_store_id,
                mcp_tools=selected_tools,
                instructions=instructions,
                force_file_search=force_file_search,
                ranker=ranker,
                retrieval_mode=retrieval_mode,
                file_search_max_chunks=file_search_max_chunks,
                file_search_score_threshold=file_search_score_threshold,
                file_search_max_tokens_per_chunk=file_search_max_tokens_per_chunk,
            )
            ragas_dataset.append({
                "id": question_id,
                "question": question,
                "answer": result.get("answer", ""),
                "contexts": result.get("contexts", []),
                "ground_truth": ground_truth,
                "tool_calls": result.get("tool_calls", []),
            })
        except Exception as e:
            logger.warning("plan_execute ragas failed for %s: %s", question_id, e)
            ragas_dataset.append({
                "id": question_id,
                "question": question,
                "answer": f"ERROR: {str(e)}",
                "contexts": [],
                "ground_truth": ground_truth,
                "error": True,
            })
    return ragas_dataset


def _generate_ragas_dataset_simple(
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
    """Simple pattern: one tools_list for all questions."""
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
                answer = _strip_think_blocks(answer)

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
