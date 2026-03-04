#!/usr/bin/env python3
"""
Minimal script to test agent.invoke_stream() against a real Llama Stack server.
Streams the response in real time instead of waiting for the full answer.

Usage:
  export LLAMA_STACK_HOST=localhost
  export LLAMA_STACK_PORT=8080
  export AGENT_VECTOR_STORE_NAME=rag-store
  export AGENT_MODEL_ID="llama-3-1-8b-w4a16/llama-3-1-8b-w4a16"

  uv run python examples/simple_invoke_stream.py "What is 2+2?"
  uv run python examples/simple_invoke_stream.py "Hello" --tools none
  uv run python examples/simple_invoke_stream.py "Use the tools" --tools all
  uv run python examples/simple_invoke_stream.py "What is 2+2?" --agent lang-graph
  uv run python examples/simple_invoke_stream.py "Query cluster" --agent lang-graph --pattern plan_execute --tools all

  # Validation is not supported with streaming; use simple_invoke.py --validate instead.
"""

import argparse
import logging
import os
import sys

import httpx
from llama_stack_client import LlamaStackClient

try:
    from dotenv import load_dotenv
    # Load .env from portazgo root (parent of examples/) so it works when run via scripts/simple_invoke.sh
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
    load_dotenv(_env_path)
    load_dotenv()  # also current working directory
except ImportError:
    pass  # optional: pip install python-dotenv for .env support

# Apply LOG_LEVEL from environment (e.g. LOG_LEVEL=DEBUG in .env)
_log_level_name = (os.environ.get("LOG_LEVEL") or "INFO").strip().upper()
logging.basicConfig(
    level=getattr(logging, _log_level_name, logging.INFO),
    format="%(levelname)s:%(name)s:%(message)s",
)

# Add src so portazgo is importable when run from repo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from portazgo import Agent, discover_mcp_tools, resolve_vector_store_id


def _env_validation_options() -> dict:
    """
    Read validation options from environment.

    Env vars:
      AGENT_VALIDATE_OUTPUT: enable validation and retry (default: false)
      AGENT_MAX_VALIDATION_RETRIES: max validation retries (default: 1)
      AGENT_VALIDATION_RULES: module:function path for custom validator, e.g. mymodule:my_validator (default: use built-in)
    """
    validate = (os.environ.get("AGENT_VALIDATE_OUTPUT") or "").strip().lower() in ("true", "1", "yes")
    try:
        max_retries = int(os.environ.get("AGENT_MAX_VALIDATION_RETRIES") or "1")
    except ValueError:
        max_retries = 1
    rules_path = (os.environ.get("AGENT_VALIDATION_RULES") or "").strip()
    validation_rules = None
    if rules_path and ":" in rules_path:
        mod_path, func_name = rules_path.rsplit(":", 1)
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            validation_rules = getattr(mod, func_name)
        except (ImportError, AttributeError) as e:
            logging.getLogger(__name__).warning("Could not load AGENT_VALIDATION_RULES=%r: %s", rules_path, e)
    return {
        "validate_output": validate,
        "max_validation_retries": max_retries,
        "validation_rules": validation_rules,
    }


def get_client() -> tuple[LlamaStackClient, str]:
    host = (os.environ.get("LLAMA_STACK_HOST") or "").strip() or "localhost"
    port = (os.environ.get("LLAMA_STACK_PORT") or "8080").strip() or "8080"
    secure = (os.environ.get("LLAMA_STACK_SECURE") or "").lower() in ("true", "1", "yes")
    if not host:
        raise SystemExit(
            "LLAMA_STACK_HOST is not set or empty. Set it in .env or export it (e.g. export LLAMA_STACK_HOST=localhost)."
        )
    base_url = f"{'https' if secure else 'http'}://{host}:{port}"
    http_client = httpx.Client(verify=False, timeout=120)
    client = LlamaStackClient(base_url=base_url, http_client=http_client)
    return client, base_url


def sanitize_question_for_server(question: str) -> str:
    """Replace ASCII apostrophe with Unicode right single quote to avoid server JSON parse errors."""
    return question.replace("'", "\u2019")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run agent.invoke_stream() against a Llama Stack server (streaming output).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "question",
        nargs="*",
        default=None,
        help="Question to ask the agent (default: What is 2+2?)",
    )
    parser.add_argument(
        "--tools",
        default="none",
        metavar="SPEC",
        help="MCP tools: none (file_search only), all, or comma-separated names (e.g. cluster-insights,tool2)",
    )
    parser.add_argument(
        "--no-apostrophe-workaround",
        action="store_true",
        help="Do not replace ASCII apostrophe (') in the question (server may return 500 if it mis-parses JSON).",
    )
    parser.add_argument(
        "--agent",
        default=os.environ.get("AGENT_TYPE", "default"),
        choices=["default", "lang-graph"],
        metavar="TYPE",
        help="Agent backend: default (Llama Stack Responses API) or lang-graph.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="(Not supported with streaming; use simple_invoke.py --validate instead.)",
    )
    parser.add_argument(
        "--pattern",
        default=os.environ.get("AGENT_PATTERN", "simple"),
        choices=["simple", "plan_execute"],
        metavar="PATTERN",
        help="Lang-graph pattern: simple or plan_execute (planner selects tools, then executor).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    val_opts = _env_validation_options()
    validate_output = getattr(args, "validate", False) or val_opts["validate_output"]
    if validate_output:
        raise SystemExit(
            "Validation is not supported with streaming. Use simple_invoke.py with --validate instead."
        )
    question = " ".join(args.question).strip() if args.question else "What is 2+2?"
    if not getattr(args, "no_apostrophe_workaround", False):
        question = sanitize_question_for_server(question)

    model_id = os.environ.get("AGENT_MODEL_ID")
    if not model_id:
        raise SystemExit(
            "AGENT_MODEL_ID is not set or empty. Set it in .env or export it."
        )
    vector_store_name = (os.environ.get("AGENT_VECTOR_STORE_NAME") or "").strip() or None
    instructions = os.environ.get("AGENT_INSTRUCTIONS") or "You are a helpful assistant. Answer concisely."
    print(f"Instructions: {instructions}")

    print("Connecting to Llama Stack...")
    try:
        client, base_url = get_client()
        print(f"URL: {base_url}")
    except SystemExit:
        raise
    try:
        vs_id = resolve_vector_store_id(client, vector_store_name)
        print(f"Vector store: {vector_store_name} (ID: {vs_id})")
    except Exception as e:
        err = str(e).lower()
        if "connection" in err or "connect" in err or "nodename" in err:
            raise SystemExit(
                f"Could not reach Llama Stack at {base_url}. "
                "Check LLAMA_STACK_HOST and LLAMA_STACK_PORT in .env, and that Llama Stack is running."
            ) from e
        raise

    mcp_tools = discover_mcp_tools(client, args.tools)
    if mcp_tools:
        print(f"MCP tools: {[t.get('server_label') for t in mcp_tools]}")
    else:
        print("MCP tools: none (file_search only)")

    print(f"Model: {model_id}")
    print(f"Agent: {args.agent}")
    if args.agent == "lang-graph":
        print(f"Pattern: {args.pattern}")
    print(f"Question: {question}\n")
    print("Answer: ", end="", flush=True)

    agent = Agent(type=args.agent)
    force_file_search = (os.environ.get("FORCE_FILE_SEARCH") or "").strip().lower() in ("true", "1", "yes")
    ranker = os.environ.get("RANKER") or "default"
    retrieval_mode = os.environ.get("RETRIEVAL_MODE") or "vector"
    file_search_max_chunks = int(os.environ.get("FILE_SEARCH_MAX_CHUNKS") or "5")
    file_search_score_threshold = float(os.environ.get("FILE_SEARCH_SCORE_THRESHOLD") or "0.7")
    file_search_max_tokens_per_chunk = int(os.environ.get("FILE_SEARCH_MAX_TOKENS_PER_CHUNK") or "512")

    result_answer = ""
    result_contexts: list = []
    result_tool_calls: list = []

    for event in agent.invoke_stream(
        client=client,
        input_text=question,
        model_id=model_id,
        vector_store_id=vs_id,
        mcp_tools=mcp_tools,
        instructions=instructions,
        force_file_search=force_file_search,
        ranker=ranker,
        retrieval_mode=retrieval_mode,
        file_search_max_chunks=file_search_max_chunks,
        file_search_score_threshold=file_search_score_threshold,
        file_search_max_tokens_per_chunk=file_search_max_tokens_per_chunk,
        strip_think_blocks=False,
        pattern=getattr(args, "pattern", "simple"),
    ):
        if event["type"] == "content_delta":
            print(event["delta"], end="", flush=True)
        elif event["type"] == "done":
            result_answer = event["answer"]
            result_contexts = event.get("contexts", [])
            result_tool_calls = event.get("tool_calls", [])

    print()  # newline after streamed answer
    print(f"\nContexts: {len(result_contexts)} chunk(s)")
    if result_contexts:
        for i, c in enumerate(result_contexts):
            preview = (c[:80] + "...") if len(c) > 80 else c
            print(f"  [{i}] {preview}")
    print(f"Tool calls: {len(result_tool_calls)} ({[t.get('tool_name') for t in result_tool_calls]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
