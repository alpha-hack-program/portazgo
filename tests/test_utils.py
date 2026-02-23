# Copyright 2025 IBM, Red Hat
# SPDX-License-Identifier: Apache-2.0

import pytest
from cohorte.utils import extract_tool_calls, serialize_for_json, strip_think_blocks


class TestStripThinkBlocks:
    def test_empty_or_none(self):
        assert strip_think_blocks("") == ""
        assert strip_think_blocks(None) is None  # type: ignore[arg-type]

    def test_no_think(self):
        assert strip_think_blocks("Hello world") == "Hello world"

    def test_single_think(self):
        text = "Before <think>internal thought</think> after"
        assert strip_think_blocks(text) == "Before  after"

    def test_multiline_think(self):
        text = "Before <think>line1\nline2</think> after"
        assert strip_think_blocks(text).strip() == "Before after"

    def test_non_string_passthrough(self):
        assert strip_think_blocks(123) == 123  # type: ignore[arg-type]


class TestSerializeForJson:
    def test_primitives(self):
        assert serialize_for_json(None) is None
        assert serialize_for_json(True) is True
        assert serialize_for_json(42) == 42
        assert serialize_for_json(3.14) == 3.14
        assert serialize_for_json("x") == "x"

    def test_dict_list(self):
        assert serialize_for_json({"a": 1}) == {"a": 1}
        assert serialize_for_json([1, 2]) == [1, 2]

    def test_object_to_dict(self):
        class C:
            def __init__(self):
                self.x = 1

        assert serialize_for_json(C()) == {"x": 1}

    def test_recursive(self):
        class C:
            def __init__(self):
                self.nested = {"a": 1}

        assert serialize_for_json(C()) == {"nested": {"a": 1}}


class TestExtractToolCalls:
    def test_empty_output(self):
        class R:
            output = []

        assert extract_tool_calls(R()) == []

    def test_file_search_call(self):
        class Result:
            text = "chunk1"

        class Item:
            type = "file_search_call"
            queries = ["q1"]
            results = [Result()]

        class R:
            output = [Item()]

        out = extract_tool_calls(R())
        assert len(out) == 1
        assert out[0]["tool_name"] == "file_search"
        assert out[0]["arguments"] == {"queries": ["q1"]}
        assert out[0]["response"] == ["chunk1"]

    def test_tool_calls_fallback(self):
        class Tc:
            __dict__ = {"tool_name": "mytool", "arguments": {}, "response": "ok"}

        class R:
            tool_calls = [Tc()]

        out = extract_tool_calls(R())
        assert len(out) == 1
        assert out[0]["tool_name"] == "mytool"
        assert out[0]["response"] == "ok"
