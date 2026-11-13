"""Tests run against a local mock server; no network and no API key needed.

    python3 -m unittest discover -s tests -v
"""
import json
import os
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm_latency as L  # noqa: E402

FIRST_DELAY = 0.15
CHUNK_DELAY = 0.02
CHUNKS = 10


class Mock(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def _sse(self, obj=None, raw=None, event=None):
        if event:
            self.wfile.write(f"event: {event}\n".encode())
        self.wfile.write(b"data: " + (raw or json.dumps(obj).encode()) + b"\n\n")
        self.wfile.flush()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        model = body["model"]
        self.server.seen.append((self.path, dict(self.headers), body))
        if model == "broken":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"upstream exploded"}}')
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        time.sleep(FIRST_DELAY)
        if self.path.endswith("/messages"):
            self._sse({"type": "message_start", "message": {"usage": {"output_tokens": 1}}}, event="message_start")
            if model == "thinker":
                self._sse({"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "hmm"}})
                time.sleep(0.2)
            for _ in range(CHUNKS):
                self._sse({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "word "}})
                time.sleep(CHUNK_DELAY)
            self._sse({"type": "message_delta", "usage": {"output_tokens": 40}})
            self._sse({"type": "message_stop"})
            return
        if model == "midstream-error":
            self._sse({"choices": [{"delta": {"content": "hi"}}]})
            self._sse({"error": {"message": "overloaded"}})
            return
        if model == "cut-off":
            self._sse({"choices": [{"delta": {"content": "hi"}}]})
            return
        for _ in range(CHUNKS):
            self._sse({"choices": [{"delta": {"content": "word "}}]})
            time.sleep(CHUNK_DELAY)
        if model != "no-usage":
            self._sse({"choices": [], "usage": {"completion_tokens": 40}})
        self._sse(raw=b"[DONE]")


class LatencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), Mock)
        cls.srv.seen = []
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def req(self, model, api="openai", base=None, **kw):
        return L.one_request(api, base or self.base + "/v1", model, "hi", 64, "sk-test", 10, kw.get("reasoning", False))

    def test_openai_stream_is_timed(self):
        r = self.req("m")
        self.assertTrue(r.ok, r.error)
        self.assertGreaterEqual(r.ttft, FIRST_DELAY)
        self.assertLess(r.ttft, FIRST_DELAY + 0.3)
        self.assertGreater(r.total, r.ttft + CHUNK_DELAY * (CHUNKS - 1) * 0.8)
        self.assertEqual(r.output_tokens, 40)
        self.assertFalse(r.tokens_estimated)
        self.assertGreater(r.tps, 0)

    def test_openai_request_shape(self):
        self.req("shape")
        path, headers, body = [s for s in self.srv.seen if s[2]["model"] == "shape"][-1]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer sk-test")
        self.assertTrue(body["stream"])
        self.assertEqual(body["stream_options"], {"include_usage": True})

    def test_tokens_are_estimated_without_usage(self):
        r = self.req("no-usage")
        self.assertTrue(r.ok, r.error)
        self.assertTrue(r.tokens_estimated)
        self.assertEqual(r.output_tokens, round(len("word ") * CHUNKS / 4))

    def test_anthropic_stream_and_request_shape(self):
        for base in (self.base, self.base + "/v1"):
            r = self.req("claude", api="anthropic", base=base)
            self.assertTrue(r.ok, r.error)
            self.assertEqual(r.output_tokens, 40)
        path, headers, body = [s for s in self.srv.seen if s[2]["model"] == "claude"][-1]
        self.assertEqual(path, "/v1/messages")
        self.assertEqual(headers["x-api-key"], "sk-test")
        self.assertIn("anthropic-version", headers)
        self.assertEqual(body["max_tokens"], 64)

    def test_reasoning_is_excluded_from_ttft_by_default(self):
        visible = self.req("thinker", api="anthropic", base=self.base)
        counted = self.req("thinker", api="anthropic", base=self.base, reasoning=True)
        self.assertGreaterEqual(visible.ttft, FIRST_DELAY + 0.2)
        self.assertLess(counted.ttft, visible.ttft - 0.1)

    def test_http_error_is_a_failure(self):
        r = self.req("broken")
        self.assertFalse(r.ok)
        self.assertEqual(r.status, 500)
        self.assertIn("upstream exploded", r.error)

    def test_midstream_error_is_a_failure(self):
        r = self.req("midstream-error")
        self.assertFalse(r.ok)
        self.assertIn("overloaded", r.error)

    def test_truncated_stream_is_a_failure(self):
        r = self.req("cut-off")
        self.assertFalse(r.ok)
        self.assertIn("terminator", r.error)

    def test_connection_refused_is_a_failure(self):
        r = L.one_request("openai", "http://127.0.0.1:9/v1", "m", "hi", 8, "k", 2, False)
        self.assertFalse(r.ok)
        self.assertTrue(r.error)

    def test_summary_and_exit_code(self):
        results = L.run("openai", self.base + "/v1", "m", "hi", 64, "k", 6, 3, 10, False)
        s = L.summarize(results)
        self.assertEqual(s["requests"], 6)
        self.assertEqual(s["success_rate"], 100.0)
        self.assertLessEqual(s["ttft_p50_s"], s["ttft_p95_s"])
        self.assertEqual(L.main(["--base-url", self.base + "/v1", "--model", "m", "--api-key", "k", "-n", "2", "--warmup", "0"]), 0)
        self.assertEqual(L.main(["--base-url", self.base + "/v1", "--model", "broken", "--api-key", "k", "-n", "2", "--warmup", "0"]), 1)

    def test_quantile(self):
        self.assertEqual(L.quantile([1, 2, 3, 4, 5], 0.5), 3)
        self.assertAlmostEqual(L.quantile([1, 2, 3, 4], 0.95), 3.85)
        self.assertIsNone(L.quantile([], 0.5))


if __name__ == "__main__":
    unittest.main()
