# OpenTelemetry request tracing and optional JSONL logs

Request timing follows the vLLM request-span boundaries: queue, Prefill to first token, and Decode
from first to last token. The first release covers ordinary generation and the native PyTorch
Mooncake Store connector. P/D and Proxy tracing are outside this release. TurboMind supports ordinary
request timing; this change does not add a new Store connector to TurboMind.

## Export traces to a local Collector

Install `lmdeploy[trace]` and set the endpoint before starting the API server:

```bash
export OTEL_SERVICE_NAME=lmdeploy-full
export OTEL_EXPORTER_OTLP_ENDPOINT=http://collector.namespace.svc:4317
export OTEL_EXPORTER_OTLP_PROTOCOL=grpc
export OTEL_RESOURCE_ATTRIBUTES="deployment.environment=prod,k8s.cluster.name=cluster,k8s.namespace.name=ns,k8s.pod.name=pod,k8s.node.name=node"
export OTEL_BSP_MAX_QUEUE_SIZE=2048
export OTEL_BSP_MAX_EXPORT_BATCH_SIZE=512
export OTEL_BSP_SCHEDULE_DELAY=1000
```

The endpoint enables tracing independently of JSONL. Both outputs can be enabled together.
OTLP/gRPC is the default; `http/protobuf` supports a Collector HTTP endpoint on port 4318.
Signal-specific `OTEL_EXPORTER_OTLP_TRACES_*` settings take precedence. The HTTP generic endpoint
gets `/v1/traces` appended by the exporter; a signal-specific endpoint must include the path.
The SDK handles TLS, certificates, export timeout and transport retries using its standard settings.
Never send JSONL to the trace pipeline. Collector persistent buffering is deployment owned.

The private provider uses ParentBased(ALWAYS_ON), a bounded BatchSpanProcessor and asynchronous
export. It respects an upstream unsampled flag; there is no additional random source sampling.
Invalid startup settings fail startup; runtime instrumentation/export failures do not replace an
inference result. No SDK work or per-token span creation is added to the engine output loop.

Native spans are `llm_request` (HTTP SERVER), `inference` (one per prompt), and `prefill`, `decode` children when their complete boundaries are known. HTTP and prompt usage overlap: choose
one level when aggregating tokens. GenAI latency attributes are seconds; OTLP timestamps are epoch
nanoseconds. Store and visible output timings are scalar attributes on the prompt span.
Input preparation is the scalar `lmdeploy.preprocess_seconds`, with no separate span.
Stage durations are not pure GPU kernel times.

Each prompt span ends at its own inference completion, not at the end of the enclosing HTTP batch.
Output observations remain separate attributes; `llm.stream.end` belongs to the HTTP span.
HTTP TTFT starts at HTTP receipt; prompt TTFT starts at engine queue admission.
Token completeness is explicit through `lmdeploy.token_counts_complete`; failed/cancelled counts
are partial. Provider initialization is serialized once per process, and the SDK owns shutdown.
Handle waiting and mean TPOT use `lmdeploy.handle_wait_seconds` and `lmdeploy.tpot_seconds`.

See [the platform extension contract](request_trace_contract.json) for exact attributes and mappings.

The platform must register `lmdeploy-full` with role `full`, map `llm_request` to `model.router`,
`inference` to the new `model.inference`, and map phase names to their corresponding `model.*`
names. The `lmdeploy.*` metadata and timing attributes require explicit platform allowlisting.
Existing vLLM/SGLang-only gateway contracts will discard these spans until extended.
This implementation does not deploy or modify a Collector or Gateway.

## Enable logging

Logging is disabled by default. Before starting the API server, set:

```bash
export LMDEPLOY_TRACE_ENABLED=1
export LMDEPLOY_TRACE_LOG_DIR=/absolute/writable/logs/instance
# Optional: 256 MiB by default, with additional UTC date rotation.
export LMDEPLOY_TRACE_LOG_MAX_MB=256
```

The directory is expanded with `expanduser`, validated as writable and absolute, and created if
missing. Flags are cached at process startup; settings take effect after restart.

Logging preserves generation settings and streaming mode. Unavailable engine timings stay null;
nonstream backend execution is never switched to internal streaming for logging.

JSONL files are written to:

```text
<root>/lmdeploy/<hostname>/<YYYYMMDDTHHMMSS.ffffffZ>.<PID>.jsonl
```

The observer handles ordinary complete SSE frames directly, with a shared bounded parser for
fragmented/coalesced/CRLF frames. It decodes JSON semantically without scanning for each possible
field name first. Per-request engine observers are bound once before generation; token counts and
status are read once per output. None of these paths retain generated text.

Files are created exclusively; collisions use numeric suffixes. Rotation follows UTC dates and
configured size. Old files remain until manually removed. JSONL bypasses native logging levels.
Native stdout/stderr and its redirection remain deployment owned. Rotation does not bound total
disk use. For example, assuming 2 KiB total JSONL per request at 2,000 requests/s gives about 354 GB
per day before filesystem overhead. This is a sizing example, not measured log volume; provision
capacity and an external cleanup policy according to actual traffic.

The API server opts in when enabled. Library `pipeline()`/`AsyncEngine` users additionally pass
`enable_request_trace=True`; setting the environment alone does not create their writer thread.
`PreprocessedRequest.trace` is explicitly typed and owned by the existing request, with no side table.

## Correlation and records

`request_id` is the validated incoming `x-request-id`, never generated or rewritten by the engine.
Missing/invalid IDs remain null in JSONL and are omitted from spans. Valid IDs are echoed through
`x-request-id`; CORS exposes that header. Request/response bodies are unchanged.
All prompts in one HTTP request share the gateway request ID. Each prompt has a distinct span ID
and a zero-based `prompt_index`. Engine/output JSONL records share that prompt's `span_id` and,
when OTLP is active, its `trace_id`. In local-only mode a local span ID distinguishes prompts.
Library calls without HTTP context do not invent a gateway request ID.

A valid W3C `traceparent` is inherited; a missing/invalid context creates one HTTP root marked
`trace.context.missing=true`. Prompt and phase spans are descendants, not extra roots.
Only a safe explicit `x-session-id` may become `session.id`; affinity headers remain untouched
and are never exported raw or locally hashed. Exceptions, bodies and credentials are not attributes.

`engine_request_finished` and `request_finished` remain per-prompt summaries, including cancelled
prepared prompts. Rejected HTTP requests have an HTTP summary without an invented prompt.

| Fields | Purpose and interpretation |
| --- | --- |
| `timestamp`, `hostname`, `pid`, `backend` | Locate the instance and corresponding native logs |
| `request_id`, `prompt_index`, `event`, `component`, `path` | Associate a prompt's engine and output records and locate its batch position |
| `status`, `http_status`, `disconnected`, `finish_reason` | Distinguish success, errors, cancellation and length limits; HTTP 200 alone is insufficient |
| `input_tokens`, `output_tokens`, `cached_tokens`, `max_output_tokens` | Distinguish long prompts, long generation, cache reuse and configured output limits |
| `engine_events_trusted` | False if native event history shrinks, resets or contains overlapping loads; affected event metrics become null instead of being replayed |
| `token_counts_complete` | True only after a successful terminal engine output; failed/cancelled counts are partial |
| `preprocess_seconds` | Preprocess entry through rendering, tokenization, multimodal processing and validation |
| `handle_wait_seconds` | Generate entry through acquisition of an engine request handle |
| `queue_seconds` | Native QUEUED through first SCHEDULED |
| `prefill_seconds` | First SCHEDULED through first observed native token boundary |
| `decode_seconds` | First through last native token boundary |
| `engine_ttft_seconds` | Native QUEUED through first token |
| `tpot_seconds` | Decode interval divided by output token count minus one; null without a usable interval or for at most one token |
| `store_lookup_seconds` | Sum of observed lookup submission-to-result intervals, including asynchronous polling waits |
| `store_matched_tokens` | Maximum newly reusable token count returned by observed Store lookups |
| `kv_load_seconds`, `kv_loaded_tokens`, `kv_load_status` | Observed external-load duration, cumulative successfully published token ranges, and pending/success/failed/cancelled outcome |
| `token_delivery_delay_seconds` | First native token timestamp through its receipt by frontend generation code |
| `max_engine_output_gap_seconds` | Largest monotonic interval between frontend receipts of nonempty engine output batches |
| `engine_timestamps` | Native wall-clock queue/schedule/first-token/last-token times for correlation with delivery and native logs |
| `http_received_unix_seconds` | HTTP entry wall clock; shared by prompts within a batch |
| `total_seconds` | This prompt's preprocessing entry through generation exit |
| `http_first_body_seconds`, `http_total_seconds` | HTTP entry through first accepted body / response exit; shared transport measurements for batch prompts |
| `first_output_seconds` | This prompt's preprocessing entry through first visible reasoning/content/tool output |
| `max_output_gap_seconds` | Largest gap between visible SSE observations; metadata and terminal markers are excluded |
| `reasoning_first_seconds`, `reasoning_last_seconds`, `content_first_seconds`, `reasoning_to_content_seconds` | Locate visible reasoning and content stages; null when unobserved |

Output token counts use the backend's newly generated token IDs, including backend stop tokens.
They can differ from API usage counts that remove stop tokens. No text is retokenized. Missing input
or cache information is null. Failed/cancelled output counts retain the observed lower bound.

These fields address observed slow-request patterns: long prompts, KV waits, engine-to-frontend
first-output gaps, and stalls after generation starts. They locate intervals without claiming a
network or GPU root cause. Output-batch gaps are not per-token ITL, and TPOT is an average.

## Timing boundaries and unknown values

Prefill/Decode include scheduling, preemption and host processing; they are not GPU kernel times.
KV lookup/load may overlap queue time, so stage durations must not be added together. API output
observations are accepted server sends, not client network receipt times.

A final-only batch containing several tokens supplies a last-token observation but no first-token
boundary. Its Prefill/Decode, TTFT and TPOT are null. A backend that already provides incremental
outputs retains the available boundaries even when the HTTP response is aggregated. Scoring-only
requests do not invent a generated token. A known single-token Decode interval is zero.
If the newest token batch lacks timing, last-token and Decode/TPOT remain unknown rather than
reusing an earlier timestamp.

Local elapsed times use monotonic clocks. Native events and token times keep the existing wall
clock for PyTorch/TurboMind interoperability. Invalid, missing or regressing intervals are null.
Cross-host comparisons and native-to-frontend delay require synchronized clocks. TurboMind token
time is a Python callback observation, while PyTorch supplies model-agent timing. Buffered output
can limit boundary precision; missing boundaries cannot diagnose an unobserved interval.

Visible stages are observed from Chat/Completions, Responses, Messages and native Generate SSE.
Nonstream JSON and hidden reasoning do not infer visible stage times. SSE lines are bounded to
1 MiB; oversized or malformed lines can leave stages unknown without changing the response body.
Records accepted in one ASGI body share its observation timestamp; parsing time is not counted as
a visible output gap. Frames use JSON semantics, including escaped keys; words inside generated
text are never classified as events.
UTF-8 decoding is explicit to avoid repeated encoding detection. Responses function-call starts
count as visible output. Engine error and
cancellation outcomes are shared with the HTTP summary, including nonstream responses with HTTP 200.

## Mooncake Store

Lookup timing starts at speculative prefetch submission when present, or at the first direct
lookup otherwise. Prefetch and admission polling share the same start marker.
The scheduler records result delivery, and the paging coordinator records
load admission through terminal progress aggregated across TP ranks. Repeated polling does not
create repeated start records. A fixed-size event cursor also deduplicates PyTorch cumulative
event snapshots, so repeated outputs cannot multiply lookup/load times or loaded-token counts.
Nonempty histories that shrink or change their processed prefix are marked untrusted, not treated
as incremental batches. Untrusted event metrics remain null for the rest of the request; token
counts and independent token timing continue. Overlapping loads receive the same treatment.
Query token matches and successfully published load tokens are
separate; a positive lookup does not establish successful cache reuse.

The native lookup client can fall back to zero on errors. `store_matched_tokens=0` therefore means
no reusable match was returned, not proof of a cache miss. Consult native Store logs for the cause.
User stop/end during a load is recorded as cancelled rather than failed. Lookup cancellation emits
a closing boundary and retains its observed duration without claiming a cache miss. Cancellation
events still depend on normal native output delivery and may be unavailable after engine loss.
A failed load keeps the existing rollback/recompute behavior and is marked failed even if later
work succeeds. Loaded-token totals count actual published ranges and may include repeated ranges.

Existing asynchronous saves remain unchanged and do not delay request completion. There is no
request-level save completion or save latency field, since background work can outlive generation.
No checksum, payload copy, forced transfer or extra Store RPC is introduced.

KV events travel with ordinary engine outputs. An engine that dies before returning output can
lose pending events; missing load results are unknown. Request cancellation, lookup cancellation,
request cleanup and worker-drain behavior retain their native ownership rules.

## Lightweight writing and failure behavior

Each request maintains fixed-size timing and counters. End summaries enter an 8192-record bounded
queue; a background thread serializes and writes up to 64 queued records per flush. Overflow and
I/O failures are counted and reported from the writer with a 60-second warning limit, keeping disk
I/O and fault-reporting logs off the request's submit path. Runtime `emit` errors are counted and
suppressed, preserving successful responses and original application exceptions. Configuration
errors still fail explicit startup validation. Write/open/flush failures trigger exponential retry
backoff from 1 to 60 seconds; queued records are dropped during backoff rather than creating one
file per record. Shutdown attempts a five-second drain.

There is no per-token disk record, per-record fsync, log retention service, query utility, OTLP
collector or alert webhook. Prompt text, generated text, reasoning text, token arrays, hashes and
KV contents are not retained in JSONL. Native request logging remains a separate feature.
Queue overflow, storage faults, blocked disks and forced termination can lose records. This is a
request-diagnosis log, not an audit trail or guaranteed durable delivery mechanism.

CPU tests run with:

```bash
python -m unittest discover -s tests/observability -v
```

They exercise actual production methods with simulated backends and no GPU dependencies. Real
model execution, Mooncake Store end-to-end behavior and timing overhead require machine testing.

Real import and browser CORS smoke checks are also provided:

```bash
pytest tests/test_lmdeploy/serve/test_request_trace_runtime.py
```

The import smoke skips when torch is unavailable and otherwise imports the real package graph.
Synthetic CPU microbenchmarks do not establish service overhead. Before serving acceptance, the
pending machine matrix is PyTorch/TurboMind × logging OFF/ON × concurrency 1/16/64, recording
TTFT, ITL, P99 and throughput under the same model, runtime and workload. All GPU overhead values
remain unverified until that matrix runs.

| Backend | Concurrency | OFF TTFT / ITL / P99 / throughput | ON TTFT / ITL / P99 / throughput |
| --- | --- | --- | --- |
| PyTorch | 1 | Unverified | Unverified |
| PyTorch | 16 | Unverified | Unverified |
| PyTorch | 64 | Unverified | Unverified |
| TurboMind | 1 | Unverified | Unverified |
| TurboMind | 16 | Unverified | Unverified |
| TurboMind | 64 | Unverified | Unverified |
