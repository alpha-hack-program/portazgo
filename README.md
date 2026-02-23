# portazgo

Pluggable agent SDK to talk to Llama Stack using different agentic frameworks. Use the **default** backend (Llama Stack Responses API) or a **lang-graph** backend (stub for future implementation).

## Installation

From source with **uv** (recommended):

```bash
cd portazgo
uv sync --extra dev
```

With optional LangGraph extra (when that backend is implemented):

```bash
uv sync --extra dev --extra langgraph
```

With pip (from source):

```bash
pip install -e .
```

For PyPI (once published):

```bash
pip install portazgo
```

## Usage

### Agent with type

```python
from portazgo import Agent

# Default: Llama Stack Responses API (same as ragas_pipeline / ragas_dataset_generator)
agent = Agent(type="default")
ragas_dataset = agent.generate_ragas_dataset(
    base_dataset=base_dataset,
    client=llama_stack_client,
    model_id="my-model",
    vector_store_id=vs_id,
    mcp_tools=mcp_tools,
    instructions="Optional system prompt",
)
```

### Single query: invoke (normal agent call)

Same parameter shape as `generate_ragas_dataset`, but for one input. The name follows **LangChain/LangGraph** (`agent.invoke(input)`):

```python
from portazgo import Agent

agent = Agent(type="default")
result = agent.invoke(
    "What is the capital of France?",
    client=llama_stack_client,
    model_id="my-model",
    vector_store_id=vs_id,
    mcp_tools=[],  # or list of MCP tool configs
    instructions="You are a helpful assistant.",
)
# result["answer"] -> str
# result["contexts"] -> list[str]  (retrieved chunks + non–file_search tool responses)
# result["tool_calls"] -> list[dict]
```

### Chat with history (e.g. chatbots)

Pass `messages` so the model sees previous turns. Each message is `{"role": "user"|"assistant"|"system", "content": str}`:

```python
history = [
    {"role": "user", "content": "My name is Alice."},
    {"role": "assistant", "content": "Nice to meet you, Alice!"},
]
result = agent.invoke(
    "What's my name?",
    client=client,
    model_id=model_id,
    vector_store_id=vs_id,
    mcp_tools=[],
    messages=history,
)
# result["answer"] can refer to the conversation (e.g. "Your name is Alice.")
```

### Streaming: invoke_stream

For real-time display (e.g. Streamlit), use `invoke_stream`. It yields events: `content_delta` (chunk of text) then `done` (final answer + contexts + tool_calls). If the backend does not support token-level streaming, the full answer is sent as one delta then `done`.

```python
for event in agent.invoke_stream(
    "Explain RAG in one sentence.",
    client=client,
    model_id=model_id,
    vector_store_id=vs_id,
    mcp_tools=[],
    messages=st.session_state.messages,  # optional history
):
    if event["type"] == "content_delta":
        print(event["delta"], end="", flush=True)
    elif event["type"] == "done":
        answer, contexts, tool_calls = event["answer"], event["contexts"], event["tool_calls"]
```

### Streamlit chat example (with history + streaming)

```python
import streamlit as st
from portazgo import Agent

# Init session state
if "messages" not in st.session_state:
    st.session_state.messages = []

agent = Agent(type="default")
# client, model_id, vector_store_id from your config (e.g. sidebar)

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Your message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full = ""
        for event in agent.invoke_stream(
            prompt,
            client=client,
            model_id=model_id,
            vector_store_id=vector_store_id,
            mcp_tools=[],
            messages=st.session_state.messages[:-1],  # history (exclude current)
        ):
            if event["type"] == "content_delta":
                full += event["delta"]
                placeholder.markdown(full + "▌")
        placeholder.markdown(full)
    st.session_state.messages.append({"role": "assistant", "content": full})
```

```python
# LangGraph backend (not yet implemented; will raise NotImplementedError)
agent = Agent(type="lang-graph")
# agent.invoke(...)  # NotImplementedError
```

### Utilities

The library also exposes helpers used by the default backend, useful for custom pipelines:

```python
from portazgo import strip_think_blocks, serialize_for_json, extract_tool_calls
```

- `strip_think_blocks(text)` – remove `<think>...</think>` blocks from model output.
- `serialize_for_json(val)` – convert objects to JSON-serializable form.
- `extract_tool_calls(response)` – extract tool calls from a Llama Stack response.

## Testing a simple invoke

**Option 1: Unit tests (no Llama Stack server)**  
Runs `invoke` against a mock client so you can confirm the API shape:

```bash
cd portazgo
uv run pytest tests/test_agent.py -v -k invoke
```

**Option 2: Real invoke against Llama Stack**  
Use the example script (requires a running Llama Stack and a vector store):

```bash
cd portazgo
export LLAMA_STACK_HOST=localhost
export LLAMA_STACK_PORT=8080
# optional: AGENT_VECTOR_STORE_NAME=rag-store, AGENT_MODEL_ID="your/model"

uv run python examples/simple_invoke.py "What is 2+2?"
```

You can pass any question as arguments; default is `"What is 2+2?"`.

**Option 3: OpenShift (oc)**  
If Llama Stack is exposed on OpenShift, use the helper script to get `APPS_DOMAIN` and run the example:

```bash
cd portazgo
./scripts/run_invoke_oc.sh "What is 2+2?"
```

The script sources `.env` (for `PROJECT`, etc.), runs `oc get ingresses.config.openshift.io cluster` for the apps domain, sets `LLAMA_STACK_HOST` to `llama-stack-demo-route-${PROJECT}.${APPS_DOMAIN}`, then runs the example with any arguments you pass.

## Development

Uses [uv](https://docs.astral.sh/uv/) for the venv and running tools. From the portazgo directory:

- **Create venv and install deps:** `make install-dev` (or `uv sync --extra dev`)
- **Lock dependencies:** `make lock` (or `uv lock`)
- **Lint:** `make lint` (ruff via `uv run`)
- **Format:** `make format`
- **Tests:** `make test` (or `uv run pytest tests`)
- **Coverage:** `make coverage`
- **Build:** `make build` (or `uv run python -m build`)

## License

Apache-2.0. See [LICENSE](LICENSE).
