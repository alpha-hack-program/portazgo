# Copyright 2025 IBM, Red Hat
# SPDX-License-Identifier: Apache-2.0

from cohorte.chats import format_history_as_prefix


def test_format_history_empty():
    assert format_history_as_prefix([]) == ""


def test_format_history_single():
    out = format_history_as_prefix([{"role": "user", "content": "Hi"}])
    assert "User: Hi" in out
    assert out.endswith("\n\n")


def test_format_history_multiple():
    out = format_history_as_prefix([
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Nice to meet you!"},
    ])
    assert "User: My name is Alice." in out
    assert "Assistant: Nice to meet you!" in out
    assert out.endswith("\n\n")


def test_format_history_system():
    out = format_history_as_prefix([{"role": "system", "content": "You are helpful."}])
    assert "System: You are helpful." in out
