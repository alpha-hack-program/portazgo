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
LangGraph-based agent implementation.

Use agent type "lang-graph" to select this backend. Supports patterns:
- simple: Delegates to default_impl.
- plan_execute: LangGraph StateGraph with planner → executor nodes.
  Planner (file_search only) selects tools; executor runs with selected MCP tools.
"""

from typing import Any, Callable, Dict, Iterator, List, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

import logging

from portazgo import default_impl
from portazgo.chats import format_history_as_prefix

PatternType = Literal["simple", "plan_execute"]

logger = logging.getLogger(__name__)


class PlanExecuteState(TypedDict, total=False):
    """State for the plan_execute LangGraph flow."""

    input_text: str
    messages: List[Dict[str, str]]
    instructions: str
    effective_input: str
    selected_tools: List[Dict[str, Any]]
    answer: str
    contexts: List[str]
    tool_calls: List[Dict[str, Any]]


def _build_plan_execute_graph(
    client: Any,
    model_id: str,
    vector_store_id: str,
    mcp_tools: List[Dict[str, Any]],
    *,
    ranker: str = "default",
    retrieval_mode: str = "vector",
    file_search_max_chunks: int = 5,
    file_search_score_threshold: float = 0.7,
    file_search_max_tokens_per_chunk: int = 512,
    strip_think_blocks: bool = True,
    validate_output: bool = False,
    max_validation_retries: int = 1,
    validation_rules: Callable[[str, str], tuple[bool, str]] | None = None,
    interrupt_before_executor: bool = False,
):
    """
    Build a LangGraph StateGraph for plan_execute: planner → executor.
    Uses default_impl._select_tools for planner, default_impl.invoke for executor.
    """
    def planner_node(state: PlanExecuteState) -> Dict[str, Any]:
        effective = state.get("effective_input") or state.get("input_text", "")
        selected = default_impl._select_tools(
            client=client,
            model_id=model_id,
            vector_store_id=vector_store_id,
            input_text=effective,
            mcp_tools=mcp_tools,
            ranker=ranker,
            retrieval_mode=retrieval_mode,
            file_search_max_chunks=file_search_max_chunks,
            file_search_score_threshold=file_search_score_threshold,
            file_search_max_tokens_per_chunk=file_search_max_tokens_per_chunk,
        )
        return {"selected_tools": selected}

    def executor_node(state: PlanExecuteState) -> Dict[str, Any]:
        result = default_impl.invoke(
            input_text=state["input_text"],
            client=client,
            model_id=model_id,
            vector_store_id=vector_store_id,
            mcp_tools=state["selected_tools"],
            messages=state.get("messages"),
            instructions=state.get("instructions", ""),
            force_file_search=False,
            ranker=ranker,
            retrieval_mode=retrieval_mode,
            file_search_max_chunks=file_search_max_chunks,
            file_search_score_threshold=file_search_score_threshold,
            file_search_max_tokens_per_chunk=file_search_max_tokens_per_chunk,
            strip_think_blocks=strip_think_blocks,
            pattern="simple",
            force_tool_use=bool(state["selected_tools"]),
            validate_output=validate_output,
            max_validation_retries=max_validation_retries,
            validation_rules=validation_rules,
        )
        return {
            "answer": result["answer"],
            "contexts": result.get("contexts", []),
            "tool_calls": result.get("tool_calls", []),
        }

    builder = StateGraph(PlanExecuteState)
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", END)

    compile_kwargs: Dict[str, Any] = {}
    if interrupt_before_executor:
        compile_kwargs["checkpointer"] = MemorySaver()
        compile_kwargs["interrupt_before"] = ["executor"]

    return builder.compile(**compile_kwargs)


def _plan_execute_invoke(
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
    validate_output: bool = False,
    max_validation_retries: int = 1,
    validation_rules: Callable[[str, str], tuple[bool, str]] | None = None,
    interrupt_before_executor: bool = False,
) -> Dict[str, Any]:
    """Plan-execute flow via LangGraph: planner selects tools → executor runs."""
    effective_input = input_text.strip()
    if messages:
        prefix = format_history_as_prefix(messages)
        effective_input = prefix + "User: " + effective_input

    graph = _build_plan_execute_graph(
        client=client,
        model_id=model_id,
        vector_store_id=vector_store_id,
        mcp_tools=mcp_tools,
        ranker=ranker,
        retrieval_mode=retrieval_mode,
        file_search_max_chunks=file_search_max_chunks,
        file_search_score_threshold=file_search_score_threshold,
        file_search_max_tokens_per_chunk=file_search_max_tokens_per_chunk,
        strip_think_blocks=strip_think_blocks,
        validate_output=validate_output,
        max_validation_retries=max_validation_retries,
        validation_rules=validation_rules,
        interrupt_before_executor=interrupt_before_executor,
    )

    initial_state: PlanExecuteState = {
        "input_text": input_text,
        "messages": messages or [],
        "instructions": instructions,
        "effective_input": effective_input,
    }

    config: Dict[str, Any] = {}
    if interrupt_before_executor:
        config["configurable"] = {"thread_id": "plan_execute"}

    result = graph.invoke(initial_state, config=config)

    return {
        "answer": result.get("answer", ""),
        "contexts": result.get("contexts", []),
        "tool_calls": result.get("tool_calls", []),
    }


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
    validate_output: bool = False,
    max_validation_retries: int = 1,
    validation_rules: Callable[[str, str], tuple[bool, str]] | None = None,
) -> Dict[str, Any]:
    """
    Resolve a single input. simple → default_impl; plan_execute → LangGraph StateGraph.
    """
    if pattern == "plan_execute" and mcp_tools:
        return _plan_execute_invoke(
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
            validate_output=validate_output,
            max_validation_retries=max_validation_retries,
            validation_rules=validation_rules,
        )

    return default_impl.invoke(
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
        pattern="simple",
        validate_output=validate_output,
        max_validation_retries=max_validation_retries,
        validation_rules=validation_rules,
    )


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
    validation_rules: Callable[[str, str], tuple[bool, str]] | None = None,
) -> Iterator[Dict[str, Any]]:
    """
    Same as invoke but yields stream events.
    simple → default_impl.invoke_stream; plan_execute → planner then stream executor.
    """
    if pattern == "plan_execute" and mcp_tools:
        effective_input = input_text.strip()
        if messages:
            prefix = format_history_as_prefix(messages)
            effective_input = prefix + "User: " + effective_input
        selected_tools = default_impl._select_tools(
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
        yield from default_impl.invoke_stream(
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
            pattern="simple",
            validate_output=validate_output,
            max_validation_retries=max_validation_retries,
            validation_rules=validation_rules,
        )
        return

    yield from default_impl.invoke_stream(
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
        pattern="simple",
        validate_output=validate_output,
        max_validation_retries=max_validation_retries,
        validation_rules=validation_rules,
    )


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
    Generate RAGAS dataset. simple → default_impl; plan_execute → LangGraph per question.
    """
    if pattern != "plan_execute" or not mcp_tools:
        return default_impl.generate_ragas_dataset(
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
            pattern="simple",
        )

    ragas_dataset: List[Dict[str, Any]] = []
    for item in base_dataset:
        question = item.get("question", "")
        question_id = item.get("id", f"q_{len(ragas_dataset) + 1}")
        ground_truth = item.get("ground_truth", "")
        try:
            result = _plan_execute_invoke(
                input_text=question,
                client=client,
                model_id=model_id,
                vector_store_id=vector_store_id,
                mcp_tools=mcp_tools,
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
