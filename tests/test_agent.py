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

    def test_lang_graph_generate_ragas_dataset_not_implemented(self):
        agent = Agent(type="lang-graph")
        with pytest.raises(NotImplementedError, match="LangGraph agent backend"):
            agent.generate_ragas_dataset(
                base_dataset=[{"question": "x"}],
                client=None,
                model_id="m",
                vector_store_id="vs",
                mcp_tools=[],
            )

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
            "What is the capital of France?",
            client=MockClient(),
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
            "Hi",
            client=MockClient(),
            model_id="m",
            vector_store_id="vs",
            mcp_tools=None,
        )
        assert result["answer"] == "ok"

    def test_lang_graph_invoke_not_implemented(self):
        agent = Agent(type="lang-graph")
        with pytest.raises(NotImplementedError, match="LangGraph agent backend"):
            agent.invoke(
                "Hi",
                client=None,
                model_id="m",
                vector_store_id="vs",
                mcp_tools=[],
            )

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
            "What is my name?",
            client=MockClient(),
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
                "Hi",
                client=MockClient(),
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

    def test_lang_graph_invoke_stream_not_implemented(self):
        agent = Agent(type="lang-graph")
        with pytest.raises(NotImplementedError, match="LangGraph agent backend"):
            list(
                agent.invoke_stream(
                    "Hi",
                    client=None,
                    model_id="m",
                    vector_store_id="vs",
                    mcp_tools=[],
                )
            )
