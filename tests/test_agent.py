# Copyright 2025 IBM, Red Hat
# SPDX-License-Identifier: Apache-2.0

import pytest
from portazgo import Agent


class TestAgent:
    def test_default_type(self):
        agent = Agent(type="default")
        assert agent.type == "default"

    def test_lang_graph_type(self):
        agent = Agent(type="lang-graph")
        assert agent.type == "lang-graph"

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Agent type must be one of"):
            Agent(type="invalid")  # type: ignore[arg-type]

    def test_default_generate_ragas_dataset_uses_client(self):
        agent = Agent(type="default")
        base = [{"id": "q1", "question": "What is 2+2?", "ground_truth": "4"}]

        class Resp:
            output_text = "4"
            output = []

        class Responses:
            def create(self, **kwargs):
                return Resp()

        class MockClient:
            responses = Responses()

        client = MockClient()
        result = agent.generate_ragas_dataset(
            base_dataset=base,
            client=client,
            model_id="test-model",
            vector_store_id="vs-1",
            mcp_tools=[],
        )
        assert len(result) == 1
        assert result[0]["id"] == "q1"
        assert result[0]["question"] == "What is 2+2?"
        assert result[0]["answer"] == "4"
        assert result[0]["ground_truth"] == "4"
        assert "contexts" in result[0]

    def test_lang_graph_generate_ragas_dataset_uses_client(self):
        agent = Agent(type="lang-graph")
        base = [{"id": "q1", "question": "What is 2+2?", "ground_truth": "4"}]

        class Resp:
            output_text = "4"
            output = []

        class Responses:
            def create(self, **kwargs):
                return Resp()

        class MockClient:
            responses = Responses()

        client = MockClient()
        result = agent.generate_ragas_dataset(
            base_dataset=base,
            client=client,
            model_id="test-model",
            vector_store_id="vs-1",
            mcp_tools=[],
        )
        assert len(result) == 1
        assert result[0]["id"] == "q1"
        assert result[0]["question"] == "What is 2+2?"
        assert result[0]["answer"] == "4"
        assert result[0]["ground_truth"] == "4"
        assert "contexts" in result[0]

    def test_invoke_returns_answer_contexts_tool_calls(self):
        agent = Agent(type="default")
        class Resp:
            output_text = "Paris"
            output = []
        class Responses:
            def create(self, **kwargs):
                return Resp()
        class MockClient:
            responses = Responses()
        result = agent.invoke(
            client=MockClient(),
            input_text="What is the capital of France?",
            model_id="m",
            vector_store_id="vs-1",
            mcp_tools=[],
        )
        assert result["answer"] == "Paris"
        assert "contexts" in result
        assert "tool_calls" in result
        assert isinstance(result["contexts"], list)
        assert isinstance(result["tool_calls"], list)

    def test_invoke_mcp_tools_default_empty(self):
        agent = Agent(type="default")
        class Resp:
            output_text = "ok"
            output = []
        class Responses:
            def create(self, **kwargs):
                return Resp()
        class MockClient:
            responses = Responses()
        result = agent.invoke(
            client=MockClient(),
            input_text="Hi",
            model_id="m",
            vector_store_id="vs",
            mcp_tools=None,
        )
        assert result["answer"] == "ok"

    def test_lang_graph_invoke_returns_answer(self):
        agent = Agent(type="lang-graph")
        class Resp:
            output_text = "Hello world"
            output = []
        class Responses:
            def create(self, **kwargs):
                return Resp()
        class MockClient:
            responses = Responses()
        result = agent.invoke(
            client=MockClient(),
            input_text="Hi",
            model_id="m",
            vector_store_id="vs",
            mcp_tools=[],
        )
        assert result["answer"] == "Hello world"
        assert "contexts" in result
        assert "tool_calls" in result

    def test_invoke_with_messages_includes_history_in_input(self):
        agent = Agent(type="default")
        seen = {}

        class Resp:
            output_text = "Alice"
            output = []

        class Responses:
            def create(self, **kwargs):
                seen["input"] = kwargs.get("input", "")
                return Resp()

        class MockClient:
            responses = Responses()

        result = agent.invoke(
            client=MockClient(),
            input_text="What is my name?",
            model_id="m",
            vector_store_id="vs",
            mcp_tools=[],
            messages=[
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Nice to meet you!"},
            ],
        )
        assert result["answer"] == "Alice"
        assert "Alice" in seen["input"]
        assert "My name is Alice" in seen["input"]
        assert "What is my name?" in seen["input"]

    def test_invoke_stream_yields_delta_then_done(self):
        agent = Agent(type="default")

        class Resp:
            output_text = "Hello world"
            output = []

        class Responses:
            def create(self, **kwargs):
                return Resp()

        class MockClient:
            responses = Responses()

        events = list(
            agent.invoke_stream(
                client=MockClient(),
                input_text="Hi",
                model_id="m",
                vector_store_id="vs",
                mcp_tools=[],
            )
        )
        assert len(events) >= 2
        assert events[0]["type"] == "content_delta"
        assert events[0]["delta"] == "Hello world"
        assert events[-1]["type"] == "done"
        assert events[-1]["answer"] == "Hello world"
        assert "contexts" in events[-1]
        assert "tool_calls" in events[-1]

    def test_lang_graph_invoke_stream_yields_delta_then_done(self):
        agent = Agent(type="lang-graph")
        class Resp:
            output_text = "Hello world"
            output = []
        class Responses:
            def create(self, **kwargs):
                return Resp()
        class MockClient:
            responses = Responses()
        events = list(
            agent.invoke_stream(
                client=MockClient(),
                input_text="Hi",
                model_id="m",
                vector_store_id="vs",
                mcp_tools=[],
            )
        )
        assert len(events) >= 2
        assert events[0]["type"] == "content_delta"
        assert events[0]["delta"] == "Hello world"
        assert events[-1]["type"] == "done"
        assert events[-1]["answer"] == "Hello world"

    def test_validate_output_default_backend(self):
        """Default backend with validate_output=True runs validator and retries on failure."""
        agent = Agent(type="default")
        call_count = [0]

        def always_pass_validator(answer: str, question: str):
            call_count[0] += 1
            return True, ""

        class Resp:
            output_text = "Valid answer here"
            output = []

        class Responses:
            def create(self, **kwargs):
                return Resp()

        class MockClient:
            responses = Responses()

        result = agent.invoke(
            client=MockClient(),
            input_text="Test",
            model_id="m",
            vector_store_id="vs",
            mcp_tools=[],
            validate_output=True,
            validation_rules=always_pass_validator,
        )
        assert result["answer"] == "Valid answer here"
        assert "contexts" in result
        assert "tool_calls" in result
        assert call_count[0] >= 1

    def test_invoke_stream_validate_output_raises(self):
        """invoke_stream with validate_output=True raises ValueError."""
        agent = Agent(type="default")
        class Resp:
            output_text = "ok"
            output = []
        class Responses:
            def create(self, **kwargs):
                return Resp()
        class MockClient:
            responses = Responses()
        with pytest.raises(ValueError, match="validate_output is not supported with invoke_stream"):
            list(
                agent.invoke_stream(
                    client=MockClient(),
                    input_text="Hi",
                    model_id="m",
                    vector_store_id="vs",
                    mcp_tools=[],
                    validate_output=True,
                )
            )

    def test_default_plan_execute_selects_tools_then_executes(self):
        """Default backend plan_execute: planner selects tools, executor runs with selected tools."""
        agent = Agent(type="default")
        mcp_tools = [
            {"type": "mcp", "server_label": "cluster-insights", "server_url": "http://x"},
            {"type": "mcp", "server_label": "compatibility-engine", "server_url": "http://y"},
        ]
        call_log = []

        class Resp:
            def __init__(self, output_text: str, output=None):
                self.output_text = output_text
                self.output = output or []

        class Responses:
            def create(self, **kwargs):
                tools = kwargs.get("tools", [])
                has_mcp = any(t.get("type") == "mcp" for t in tools)
                call_log.append(has_mcp)
                if has_mcp:
                    return Resp("Executor answer")
                return Resp('{"tools": ["cluster-insights"]}')

        class MockClient:
            responses = Responses()

        result = agent.invoke(
            client=MockClient(),
            input_text="Query cluster status",
            model_id="m",
            vector_store_id="vs",
            mcp_tools=mcp_tools,
            pattern="plan_execute",
        )
        assert result["answer"] == "Executor answer"
        assert len(call_log) == 2
        assert call_log[0] is False
        assert call_log[1] is True

    def test_lang_graph_plan_execute_selects_tools_then_executes(self):
        """Lang-graph plan_execute: delegates to default_impl, same behavior."""
        agent = Agent(type="lang-graph")
        mcp_tools = [
            {"type": "mcp", "server_label": "cluster-insights", "server_url": "http://x"},
            {"type": "mcp", "server_label": "compatibility-engine", "server_url": "http://y"},
        ]
        call_log = []

        class Resp:
            def __init__(self, output_text: str, output=None):
                self.output_text = output_text
                self.output = output or []

        class Responses:
            def create(self, **kwargs):
                tools = kwargs.get("tools", [])
                has_mcp = any(t.get("type") == "mcp" for t in tools)
                call_log.append(has_mcp)
                if has_mcp:
                    return Resp("Executor answer")
                return Resp('{"tools": ["cluster-insights"]}')

        class MockClient:
            responses = Responses()

        result = agent.invoke(
            client=MockClient(),
            input_text="Query cluster status",
            model_id="m",
            vector_store_id="vs",
            mcp_tools=mcp_tools,
            pattern="plan_execute",
        )
        assert result["answer"] == "Executor answer"
        assert len(call_log) == 2
        assert call_log[0] is False
        assert call_log[1] is True

    def test_validate_output_lang_graph_backend(self):
        """Lang-graph backend with validate_output=True runs validator (same logic as default)."""
        agent = Agent(type="lang-graph")
        call_count = [0]

        def always_pass_validator(answer: str, question: str):
            call_count[0] += 1
            return True, ""

        class Resp:
            output_text = "Valid answer here"
            output = []

        class Responses:
            def create(self, **kwargs):
                return Resp()

        class MockClient:
            responses = Responses()

        result = agent.invoke(
            client=MockClient(),
            input_text="Test",
            model_id="m",
            vector_store_id="vs",
            mcp_tools=[],
            validate_output=True,
            validation_rules=always_pass_validator,
        )
        assert result["answer"] == "Valid answer here"
        assert call_count[0] >= 1
