"""Real SDK, protobuf transport, ASGI context and privacy tests; no GPU required."""
import asyncio
import json
import os
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

from test_request_trace import SOURCE, obs, otel

try:
    import grpc
    from opentelemetry.proto.collector.trace.v1 import trace_service_pb2 as pb
    from opentelemetry.proto.collector.trace.v1 import trace_service_pb2_grpc as rpc
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.sdk.trace.sampling import ALWAYS_ON, ParentBased
    SDK = True
except ImportError:
    SDK = False


@unittest.skipUnless(SDK, 'install lmdeploy[trace] to test SDK and OTLP wire transport')
class OTLPTests(unittest.TestCase):
    def setUp(self):
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider(sampler=ParentBased(ALWAYS_ON))
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.patch = patch.object(otel, '_tracer', self.provider.get_tracer('test'))
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.addCleanup(self.provider.shutdown)

    def request(self, headers=(), status=200, count=2, crash=False):
        records, sent = [], []
        async def app(scope, receive, send):
            for _ in range(count):
                t = obs.RequestTrace()
                t.model = 'test-model'
                t.backend = 'turbomind'
                t.terminal = True
                t.prompt_tokens, t.generated_tokens, t.cached_tokens = 10, 2, 0
                t.preprocessed = t.start + .00001
                stamp = t.start_ns / 1e9
                t.queued, t.scheduled = stamp + .00002, stamp + .00003
                t.first_token, t.last_token = stamp + .00004, stamp + .00005
                time.sleep(.001)
                t.finish('error' if crash else 'success')
            if crash:
                raise ValueError('Bearer private-completion')
            await send({'type': 'http.response.start', 'status': status, 'headers': []})
            await send({'type': 'http.response.body', 'body': b'private-completion'})
        async def receive():
            return {'type': 'http.request', 'body': b'private-prompt'}
        async def send(message):
            sent.append(message)
        def record(event, rid, **kw):
            records.append(dict(event=event, request_id=rid, **kw))

        async def run():
            with patch.object(obs, 'enabled', return_value=True), patch.object(obs, 'get_writer'), \
                 patch.object(obs, 'emit', side_effect=record):
                await obs.RequestTraceMiddleware(app)(
                    {'type': 'http', 'method': 'POST', 'path': '/v1/completions', 'headers': headers}, receive, send)
        if crash:
            with self.assertRaisesRegex(ValueError, 'private-completion'):
                asyncio.run(run())
        else:
            asyncio.run(run())
        self.assertIsNone(obs.request_outcome_context.get())
        return records, sent

    def test_parent_prompt_stages_and_readonly_ids(self):
        trace_id, parent_id = 'a'*32, 'b'*16
        records, sent = self.request([(b'traceparent', f'00-{trace_id}-{parent_id}-01'.encode()),
            (b'x-request-id', b'req_gateway_123456'), (b'x-session-affinity', b'private-affinity')])
        spans = self.exporter.get_finished_spans()
        root = next(s for s in spans if s.name == 'llm_request')
        prompts = [s for s in spans if s.name == 'inference']
        self.assertEqual(len(spans), 7)
        self.assertEqual(root.parent.span_id, int(parent_id, 16))
        self.assertEqual(root.context.trace_id, int(trace_id, 16))
        self.assertEqual(root.attributes['gen_ai.usage.output_tokens'], 4)
        self.assertTrue(all(p.parent.span_id == root.context.span_id for p in prompts))
        self.assertEqual(len({p.context.span_id for p in prompts}), 2)
        for s in spans:
            self.assertLessEqual(s.start_time, s.end_time)
            if s.name in ('prefill', 'decode'):
                self.assertIn(s.parent.span_id, {p.context.span_id for p in prompts})
        self.assertTrue(all(r['request_id'] == 'req_gateway_123456' for r in records))
        self.assertEqual(len({r['span_id'] for r in records}), 2)
        self.assertEqual(sent[-1]['body'], b'private-completion')
        wire = str([dict(s.attributes) for s in spans])
        self.assertNotIn('private', wire)

    def test_missing_invalid_context_and_ids(self):
        self.request([(b'traceparent', b'invalid'), (b'x-request-id', b'../invalid')], count=0, status=400)
        root = self.exporter.get_finished_spans()[0]
        self.assertIsNone(root.parent)
        self.assertTrue(root.attributes['trace.context.missing'])
        self.assertNotIn('request.id', root.attributes)
        self.assertEqual(root.status.status_code.name, 'ERROR')
        self.assertEqual(root.attributes['http.response.status_code'], 400)

    def test_unsampled_parent_and_exception_privacy(self):
        self.request([(b'traceparent', ('00-'+'a'*32+'-'+'b'*16+'-00').encode())])
        self.assertEqual(len(self.exporter.get_finished_spans()), 0)
        self.request(crash=True)
        spans = self.exporter.get_finished_spans()
        self.assertTrue(all(s.status.status_code.name == 'ERROR' for s in spans))
        self.assertNotIn('private-completion', str(spans))

    def test_instrumentation_failure_does_not_replace_result(self):
        with patch.object(otel._tracer, 'start_span', side_effect=RuntimeError('private')):
            _, sent = self.request()
        self.assertEqual(sent[-1]['body'], b'private-completion')

    def test_batch_prompt_ends_before_next_prompt_and_error_events_in_bounds(self):
        self.request()
        spans = self.exporter.get_finished_spans()
        prompts = sorted((s for s in spans if s.name == 'inference'), key=lambda s: s.start_time)
        self.assertLessEqual(prompts[0].end_time, prompts[1].start_time)
        root = next(s for s in spans if s.name == 'llm_request')
        self.assertGreater(root.attributes['gen_ai.latency.time_to_first_token'], 0)
        self.exporter.clear()
        self.request(crash=True)
        for span in self.exporter.get_finished_spans():
            for event in span.events:
                self.assertLessEqual(span.start_time, event.timestamp)
                self.assertLessEqual(event.timestamp, span.end_time)

    def test_failed_http_span_does_not_create_prompt_roots(self):
        original = otel._tracer.start_span
        def start(name, **kwargs):
            if name == 'llm_request':
                raise RuntimeError('failure')
            return original(name, **kwargs)
        with patch.object(otel._tracer, 'start_span', side_effect=start):
            self.request()
        self.assertEqual(len(self.exporter.get_finished_spans()), 0)

    def test_scalar_attributes_and_context_cleanup_after_bad_metadata(self):
        span = otel._tracer.start_span('inference')
        otel.attributes(span, {'safe': 1, 'array': ['private'], 'object': {'private': 1}, 'nan': float('nan')})
        otel.finish(span, 'success')
        self.assertNotIn('array', self.exporter.get_finished_spans()[0].attributes)
        self.assertNotIn('object', self.exporter.get_finished_spans()[0].attributes)
        original = obs.RequestTrace.finish
        def finish(t, status):
            original(t, status)
            t.model = []  # Deliberately break HTTP model aggregation.
        with patch.object(obs.RequestTrace, 'finish', finish):
            _, sent = self.request()
        self.assertEqual(sent[-1]['body'], b'private-completion')
        self.assertIsNone(obs.request_outcome_context.get())

    def test_emitted_fields_match_platform_extension_contract(self):
        contract = json.loads((SOURCE / 'docs/en/serving/request_trace_contract.json').read_text())
        self.request([(b'x-request-id', b'req_contract_123456')])
        for span in self.exporter.get_finished_spans():
            self.assertIn(span.name, contract['span_mapping'])
            self.assertTrue(set(span.attributes) <= set(contract['allowed_attributes']))
            self.assertTrue(all(isinstance(v, (str, int, bool, float)) for v in span.attributes.values()))
            self.assertTrue(all(e.name in contract['events'] for e in span.events))
        prompt = next(s for s in self.exporter.get_finished_spans() if s.name == 'inference')
        self.assertGreaterEqual(prompt.attributes['lmdeploy.tpot_seconds'], 0)
        self.assertGreaterEqual(prompt.attributes['lmdeploy.preprocess_seconds'], 0)
        self.assertFalse(any(s.name == 'tokenize' for s in self.exporter.get_finished_spans()))

    def test_concurrent_initialization_creates_one_provider(self):
        calls = []
        def initialize():
            time.sleep(.01)
            calls.append(1)
            otel._tracer = object()
            otel._pid = os.getpid()
        with patch.object(otel, 'ENDPOINT', 'http://unused'), patch.object(otel, '_pid', None), \
                patch.object(otel, '_tracer', None), patch.object(otel, '_initialize', side_effect=initialize):
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda _: otel.initialize(), range(32)))
        self.assertEqual(len(calls), 1)

    def test_token_completeness_is_explicit_for_success_and_failure(self):
        self.request()
        for span in self.exporter.get_finished_spans():
            if span.name in ('inference', 'llm_request'):
                self.assertTrue(span.attributes['lmdeploy.token_counts_complete'])
        self.exporter.clear()
        self.request(crash=True)
        for span in self.exporter.get_finished_spans():
            if span.name in ('inference', 'llm_request'):
                self.assertFalse(span.attributes['lmdeploy.token_counts_complete'])

    def test_export_queue_does_not_wait_for_exporter(self):
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
        entered, release = threading.Event(), threading.Event()
        class BlockedExporter(SpanExporter):
            def export(self, spans):
                entered.set()
                release.wait(5)
                return SpanExportResult.FAILURE
            def shutdown(self):
                pass
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(BlockedExporter(),
            max_queue_size=8, max_export_batch_size=1, schedule_delay_millis=1))
        try:
            with patch.object(otel, '_tracer', provider.get_tracer('test')):
                self.request(count=1)
                self.assertTrue(entered.wait(1))
                start = time.monotonic()
                for _ in range(4):
                    self.request(count=1)
                self.assertLess(time.monotonic() - start, 1)
        finally:
            release.set()
            provider.shutdown()

    def test_sensitive_metadata_is_not_exported(self):
        self.request([(b'x-request-id', b'sk-private_12345678'),
                      (b'x-session-id', b'sk-ant-private')], count=0)
        root = self.exporter.get_finished_spans()[0]
        self.assertNotIn('request.id', root.attributes)
        self.assertNotIn('session.id', root.attributes)

    def test_real_grpc_and_http_protobuf_batch_export(self):
        for protocol in ('grpc', 'http/protobuf'):
            received = []
            if protocol == 'grpc':
                class Receiver(rpc.TraceServiceServicer):
                    def Export(self, request, context):
                        received.append(request)
                        return pb.ExportTraceServiceResponse()
                server = grpc.server(ThreadPoolExecutor(max_workers=1))
                rpc.add_TraceServiceServicer_to_server(Receiver(), server)
                port = server.add_insecure_port('127.0.0.1:0')
                server.start()
                def stop():
                    server.stop(0).wait()
            else:
                class Handler(BaseHTTPRequestHandler):
                    def do_POST(self):
                        self.server.test.assertEqual(self.path, '/v1/traces')
                        self.server.test.assertEqual(self.headers['Content-Type'], 'application/x-protobuf')
                        received.append(pb.ExportTraceServiceRequest.FromString(self.rfile.read(int(self.headers['Content-Length']))))
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/x-protobuf')
                        self.end_headers()
                        self.wfile.write(pb.ExportTraceServiceResponse().SerializeToString())
                    def log_message(self, *args):
                        pass
                server = HTTPServer(('127.0.0.1', 0), Handler)
                server.test = self
                port = server.server_port
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                def stop():
                    server.shutdown()
                    server.server_close()
                    thread.join()
            try:
                endpoint = f'http://127.0.0.1:{port}'
                with patch.dict(os.environ, {'OTEL_EXPORTER_OTLP_ENDPOINT': endpoint,
                     'OTEL_EXPORTER_OTLP_PROTOCOL': protocol, 'OTEL_SERVICE_NAME': 'lmdeploy-full',
                     'NO_PROXY': '*', 'no_proxy': '*'}, clear=True), \
                     patch.object(otel, 'ENDPOINT', endpoint), patch.object(otel, '_pid', None):
                    otel.initialize()
                    provider = otel._provider
                    try:
                        self.request(count=1)
                        self.assertTrue(provider.force_flush(5000))
                        spans = [s for req in received for rs in req.resource_spans
                                 for ss in rs.scope_spans for s in ss.spans]
                        self.assertEqual(len(spans), 4)
                        self.assertTrue(all(len(s.trace_id) == 16 and len(s.span_id) == 8 for s in spans))
                    finally:
                        provider.shutdown()
                        otel._tracer = self.provider.get_tracer('test')
            finally:
                stop()
