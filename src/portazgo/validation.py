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

"""Shared validation logic for agent answers."""

# Default phrases that indicate an unsatisfactory answer (rule-based validation)
DEFAULT_REJECT_PHRASES = (
    "i don't know",
    "i cannot",
    "i can't",
    "i'm unable",
    "i am unable",
    "no information",
    "not found",
    "cannot find",
    "could not find",
    "unable to find",
)


def default_validator(answer: str, question: str) -> tuple[bool, str]:
    """
    Rule-based validation: reject empty, very short, or unsatisfactory answers.

    Returns (passed, feedback). If passed is False, feedback explains why.
    """
    if not answer or not isinstance(answer, str):
        return False, "Answer is empty."
    text = answer.strip()
    if len(text) < 10:
        return False, "Answer is too short."
    lower = text.lower()
    for phrase in DEFAULT_REJECT_PHRASES:
        if phrase in lower:
            return False, f"Answer contains unsatisfactory phrase: {phrase!r}"
    return True, ""
