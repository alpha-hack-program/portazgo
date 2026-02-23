#!/usr/bin/env python3
"""
Minimal script to test agent.invoke() against a real Llama Stack server.

Usage:
  export LLAMA_STACK_HOST=localhost
  export LLAMA_STACK_PORT=8080
  export AGENT_VECTOR_STORE_NAME=rag-store
  export AGENT_MODEL_ID="llama-3-1-8b-w4a16/llama-3-1-8b-w4a16"

  uv run python examples/simple_invoke.py "What is 2+2?"
  uv run python examples/simple_invoke.py "Hello" --tools none
  uv run python examples/simple_invoke.py "Use the tools" --tools all
  uv run python examples/simple_invoke.py "Query cluster" --tools cluster-insights,compatibility-engine
"""

import argparse
import logging
import os
import sys
from typing import Any

import httpx
from llama_stack_client import LlamaStackClient

try:
    from dotenv import load_dotenv
    # Load .env from cohorte root (parent of examples/) so it works when run via scripts/simple_invoke.sh
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

# Add src so cohorte is importable when run from repo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cohorte import Agent


def discover_mcp_tools(client: LlamaStackClient, tools: str) -> list[dict[str, Any]]:
    """Discover MCP tools from Llama Stack. tools: 'none' (default), 'all', or 'tool1,tool2'."""
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
            server_url = getattr(mcp_endpoint, "uri", None) or (
                mcp_endpoint.get("uri") if isinstance(mcp_endpoint, dict) else None
            )
        if server_url:
            mcp_tools.append({
                "type": "mcp",
                "server_label": tool_name,
                "server_url": server_url,
            })
    return mcp_tools


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


def get_vector_store_id(client: LlamaStackClient, vector_store_name: str | None) -> str:
    stores = list(client.vector_stores.list())
    if not stores:
        raise SystemExit("No vector stores found. Create one first.")
    if vector_store_name:
        matching = [s for s in stores if getattr(s, "name", None) == vector_store_name]
        if not matching:
            raise SystemExit(
                f"No vector store named {vector_store_name!r}. "
                f"Available: {[getattr(s, 'name', None) for s in stores]}"
            )
        return matching[0].id
    return stores[-1].id


def sanitize_question_for_server(question: str) -> str:
    """Replace ASCII apostrophe with Unicode right single quote to avoid server JSON parse errors.

    Some servers build JSON with the user input and treat ASCII apostrophe (') as a string
    terminator, causing 'Unterminated string' (e.g. in \"I'm\"). Using U+2019 (') avoids that.
    """
    return question.replace("'", "\u2019")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run agent.invoke() against a Llama Stack server.",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
        vs_id = get_vector_store_id(client, vector_store_name)
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
    print(f"Question: {question}\n")

    # Initialize the agent
    agent = Agent(type="default")
    force_file_search = (os.environ.get("FORCE_FILE_SEARCH") or "").strip().lower() in ("true", "1", "yes")
    ranker = os.environ.get("RANKER") or "default"
    retrieval_mode = os.environ.get("RETRIEVAL_MODE") or "vector"
    file_search_max_chunks = int(os.environ.get("FILE_SEARCH_MAX_CHUNKS") or "5")
    file_search_score_threshold = float(os.environ.get("FILE_SEARCH_SCORE_THRESHOLD") or "0.7")
    file_search_max_tokens_per_chunk = int(os.environ.get("FILE_SEARCH_MAX_TOKENS_PER_CHUNK") or "512")

    # Make a call with all the tools
    result = agent.invoke(
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
    )

    print("Answer:", result["answer"])
    if result.get("contexts"):
        print(f"Contexts: {len(result['contexts'])} chunk(s)")
        # Print the first 50 characters of each context
        print(f"Contexts: {[c[:50] + '...' if len(c) > 50 else c for c in result['contexts']]}")
    if result.get("tool_calls"):
        print("Tool calls:", [t.get("tool_name") for t in result["tool_calls"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
