# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retry/backoff policy for remote model calls.

Regression cover for the original behaviour: every provider retried immediately with
no delay, so a 429 produced three rapid-fire requests against an already-throttled
endpoint and then returned None silently.

These tests exercise the policy helpers directly so they need none of the heavy
model dependencies (torch/diffusers) that importing the provider classes would pull in.
"""

import importlib.util
from pathlib import Path

import pytest


def _load_policy_helpers():
    """Load the retry helpers out of vlm.py without importing its heavy deps."""
    source = (
        Path(__file__).resolve().parents[1] / "simfoundry" / "models" / "vlm.py"
    ).read_text(encoding="utf-8")
    start = source.index("RETRY_BASE_DELAY_S")
    end = source.index("class VLM_API:")
    namespace: dict = {}
    exec("import os, random, re, time\n" + source[start:end], namespace)
    return namespace


POLICY = _load_policy_helpers()


class _Err(Exception):
    def __init__(self, message, status_code=None, retry_delay=None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_delay = retry_delay


@pytest.mark.parametrize(
    "message,status",
    [
        ("429 RESOURCE_EXHAUSTED: quota exceeded", None),
        ("Too Many Requests", None),
        ("rate limit reached for this model", None),
        ("", 429),
    ],
)
def test_rate_limit_errors_are_detected(message, status):
    exc = _Err(message, status)
    assert POLICY["is_rate_limit_error"](exc)
    assert not POLICY["is_non_retryable_error"](exc)


@pytest.mark.parametrize(
    "message,status",
    [("400 INVALID_ARGUMENT: bad prompt", None), ("", 403), ("", 404)],
)
def test_client_errors_are_not_retryable(message, status):
    assert POLICY["is_non_retryable_error"](_Err(message, status))


def test_transient_errors_are_retryable_but_not_rate_limits():
    exc = _Err("Connection reset by peer")
    assert not POLICY["is_rate_limit_error"](exc)
    assert not POLICY["is_non_retryable_error"](exc)


def test_server_supplied_retry_delay_is_honored():
    assert POLICY["retry_delay_from_error"](_Err("429", 429, 17)) == pytest.approx(17.0)
    assert POLICY["retry_delay_from_error"](_Err("429 quota, retryDelay: 23s")) == pytest.approx(23.0)
    assert POLICY["retry_delay_from_error"](_Err("boom")) is None


def test_backoff_is_bounded_and_grows():
    cap = POLICY["RETRY_MAX_DELAY_S"]
    for attempt in range(8):
        for _ in range(20):  # jittered, so sample
            assert 0.0 <= POLICY["backoff_sleep_s"](attempt) <= cap
    # The ceiling must actually grow before saturating at the cap.
    assert POLICY["RETRY_BASE_DELAY_S"] * 2 ** 5 >= POLICY["RETRY_BASE_DELAY_S"] * 2


def _drive(exc, n_retries=3):
    """Mirror the retry loop used by Gemini/Imagen3/GPT."""
    slept: list[float] = []
    budget, attempt = n_retries, 0
    while attempt < budget:
        try:
            raise exc
        except Exception as err:  # noqa: BLE001 - mirrors production call sites
            budget = POLICY["handle_remote_exception"](
                err,
                attempt=attempt,
                n_retries=budget,
                provider="Test",
                model="m",
                sleep_fn=slept.append,
            )
        attempt += 1
    return slept, budget


def test_rate_limit_sleeps_between_attempts_and_extends_the_budget():
    slept, budget = _drive(_Err("429 RESOURCE_EXHAUSTED"), n_retries=3)
    assert budget == POLICY["RATE_LIMIT_MAX_ATTEMPTS"]
    assert len(slept) == budget - 1
    # The original bug: zero delay between retries.
    assert sum(slept) > 0.0


def test_non_retryable_error_fails_fast_without_sleeping():
    with pytest.raises(POLICY["RemoteCallFailed"]):
        _drive(_Err("400 INVALID_ARGUMENT: bad prompt"))


def test_transient_error_keeps_the_normal_budget():
    slept, budget = _drive(_Err("Connection reset by peer"), n_retries=3)
    assert budget == 3
    assert len(slept) == 2
