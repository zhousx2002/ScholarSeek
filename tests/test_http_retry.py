import json
import http.client
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request

from scholarseek.http_retry import request_json


class _FakeGate:
    def __init__(self):
        self.wait_count = 0
        self.delays = []
        self.blocks = []

    def wait(self):
        self.wait_count += 1

    def defer(self, seconds):
        self.delays.append(seconds)

    def block(self, seconds):
        self.blocks.append(seconds)


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class HttpRetryTest(unittest.TestCase):
    def test_long_retry_after_opens_circuit_without_sleeping(self):
        rate_limit = HTTPError(
            "https://example.com",
            429,
            "Too Many Requests",
            {"Retry-After": "120"},
            None,
        )
        gate = _FakeGate()
        with patch("scholarseek.http_retry.urlopen", side_effect=rate_limit) as mocked:
            with self.assertRaises(RuntimeError):
                request_json(
                    Request("https://example.com"),
                    service="test",
                    gate=gate,
                    timeout=1,
                    max_retries=4,
                )

        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(gate.blocks, [120.0])

    def test_retries_rate_limit_and_respects_retry_after(self):
        rate_limit = HTTPError(
            "https://example.com",
            429,
            "Too Many Requests",
            {"Retry-After": "3"},
            None,
        )
        gate = _FakeGate()
        with patch("scholarseek.http_retry.urlopen", side_effect=[rate_limit, _FakeResponse({"ok": True})]):
            payload = request_json(
                Request("https://example.com"),
                service="test",
                gate=gate,
                timeout=1,
                max_retries=1,
            )

        self.assertEqual(payload, {"ok": True})
        self.assertEqual(gate.wait_count, 2)
        self.assertEqual(gate.delays, [3.0])

    def test_does_not_retry_non_retryable_http_error(self):
        bad_request = HTTPError("https://example.com", 400, "Bad Request", {}, None)
        gate = _FakeGate()
        with patch("scholarseek.http_retry.urlopen", side_effect=bad_request) as mocked:
            with self.assertRaises(RuntimeError):
                request_json(
                    Request("https://example.com"),
                    service="test",
                    gate=gate,
                    timeout=1,
                    max_retries=4,
                )

        self.assertEqual(mocked.call_count, 1)

    def test_retries_remote_disconnect(self):
        gate = _FakeGate()
        with patch(
            "scholarseek.http_retry.urlopen",
            side_effect=[
                http.client.RemoteDisconnected("closed"),
                _FakeResponse({"ok": True}),
            ],
        ):
            payload = request_json(
                Request("https://example.com"),
                service="test",
                gate=gate,
                timeout=1,
                max_retries=1,
            )

        self.assertEqual(payload, {"ok": True})
        self.assertEqual(gate.wait_count, 2)
        self.assertEqual(gate.delays, [1.0])


if __name__ == "__main__":
    unittest.main()
