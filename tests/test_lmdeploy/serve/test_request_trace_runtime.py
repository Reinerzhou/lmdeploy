# Copyright (c) OpenMMLab. All rights reserved.
"""Real import and CORS smoke tests, requiring the installed serving environment."""
import asyncio
import importlib

import pytest

pytest.importorskip('torch')


def test_trace_real_import_graph():
    for module in (
        'lmdeploy.trace',
        'lmdeploy.serve.core.async_engine',
        'lmdeploy.serve.openai.api_server',
        'lmdeploy.pytorch.kv_connector.mooncake.store.scheduler',
        'lmdeploy.pytorch.paging.kv_load_coordinator',
    ):
        importlib.import_module(module)


def test_outer_trace_header_is_exposed_to_browser(monkeypatch, tmp_path):
    pytest.importorskip('httpx')
    from httpx import ASGITransport, AsyncClient
    from starlette.applications import Starlette
    from starlette.middleware.cors import CORSMiddleware
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    from lmdeploy.serve.observability import request_trace

    monkeypatch.setattr(request_trace, 'ENABLED', True)
    writer = request_trace.JsonlWriter(tmp_path)
    monkeypatch.setattr(request_trace, 'get_writer', lambda: writer)

    async def endpoint(request):
        return JSONResponse({'ok': True})

    app = Starlette(routes=[Route('/v1/completions', endpoint, methods=['POST'])])
    app.add_middleware(CORSMiddleware, allow_origins=['https://client.example'],
                       expose_headers=[request_trace.HEADER])
    app.add_middleware(request_trace.RequestTraceMiddleware)
    try:
        async def request():
            async with AsyncClient(transport=ASGITransport(app),
                                   base_url='http://testserver', trust_env=False) as client:
                return await client.post('/v1/completions',
                                         headers={'Origin': 'https://client.example',
                                                  'x-request-id': 'req_browser_123456'})
        response = asyncio.run(request())
        assert response.status_code == 200
        assert request_trace.HEADER in response.headers
        exposed = response.headers['access-control-expose-headers'].lower()
        assert request_trace.HEADER in exposed
    finally:
        writer.close()
