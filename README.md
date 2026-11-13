# llm-latency

Measure time to first token, total latency, output speed and success rate of any OpenAI-compatible or Anthropic-native streaming endpoint, from where your code actually runs.

One file, standard library only, Python 3.9+.

```bash
export LLM_API_KEY=sk-...
python3 llm_latency.py --base-url https://api.openai.com/v1 --model gpt-6.1-sol -n 20
```

```
model        n  success  ttft p50  ttft p95  total p50  total p95  tok/s p50
-----------  -  -------  --------  --------  ---------  ---------  ---------
gpt-6.1-sol  20  100.0%   0.612 s   1.904 s    4.310 s    6.722 s       58.3
```

(The numbers above are illustrative, not a measurement.)

## Why

Vendor status pages tell you whether an API is up. They do not tell you how long your users wait for the first token from your region, through your proxy, with your prompt. This script answers that with a few dozen real requests, and gives you percentiles instead of one average.

## Usage

Compare several models on the same endpoint:

```bash
python3 llm_latency.py --base-url https://api.openai.com/v1 \
  --model gpt-6.1-sol --model gpt-6-luna -n 30 -c 3
```

Anthropic Messages API:

```bash
python3 llm_latency.py --api anthropic --base-url https://api.anthropic.com \
  --model claude-opus-5-5 -n 20
```

Any gateway or self-hosted server that speaks one of the two formats works the same way: point `--base-url` at it.

Use your own prompt and keep the raw data:

```bash
python3 llm_latency.py --base-url https://api.openai.com/v1 --model gpt-6.1-sol \
  --prompt-file prompt.txt --max-tokens 1024 --json run.json --csv run.csv
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--base-url` | required | Up to and including `/v1` for OpenAI-style endpoints; the host root for Anthropic |
| `--model` | required | Model ID; repeat to compare |
| `--api` | `openai` | `openai` (Chat Completions) or `anthropic` (Messages) |
| `--api-key` | `$LLM_API_KEY` | Never logged or written to the output files |
| `-n` | 20 | Measured requests per model |
| `-c` | 1 | Parallel requests |
| `--warmup` | 1 | Unmeasured requests first, so connection setup and cold caches do not skew the run |
| `--max-tokens` | 512 | Output cap per request |
| `--timeout` | 120 | Socket timeout in seconds |
| `--count-reasoning` | off | Treat streamed reasoning text as the first token |
| `--json`, `--csv` | off | Write results to a file |

The exit code is 0 only when every request succeeded, so the script can gate a deploy or run from cron.

## What the numbers mean

- **ttft**: seconds from sending the request to the first streamed content token. For reasoning models this includes thinking time, because that is what the user waits for. Pass `--count-reasoning` to stop the clock at the first reasoning token instead.
- **total**: seconds from sending the request to the end of the stream.
- **tok/s**: output tokens divided by generation time (`total - ttft`). Token counts come from the `usage` block when the server sends one. When it does not, tokens are estimated as characters / 4 and the column is prefixed with `~`.
- **success**: HTTP 2xx, at least one content token, and a proper end of stream. A stream that is cut off without `[DONE]` or `message_stop`, or that carries an error event, counts as a failure even though the status was 200.

## Reading the results honestly

- Twenty requests give you a usable p50 and a rough p95. For a p95 you would act on, run a few hundred.
- Results depend on the prompt, `max_tokens`, time of day and your network. Compare models with the same prompt, in the same run.
- With `-c` above 1 you are also measuring rate limits and queueing. That is useful, but it is a different number.
- Every request is billed by your provider. The default run is small on purpose.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The tests start a local mock server. They need no network and no API key.

## About

I work on [byesu](https://byesu.com), an LLM API gateway, and wrote this to check latency from the client side. The gateway publishes [weekly per-model latency and success rates](https://byesu.com/benchmarks) measured on its own traffic; this script lets you check any endpoint, including that one, yourself.

MIT licensed.
