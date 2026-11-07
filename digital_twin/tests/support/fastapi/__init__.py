"""
Minimal FastAPI stand-in, used ONLY by the test suite when the real `fastapi`
package is not installed (it could not be installed in the environment where
this build was tested). It implements the small subset of FastAPI that
backend/main.py uses - GET/POST routes, query/path parameters, Pydantic request
bodies, HTTPException, startup events, CORS middleware and StaticFiles mounts -
on top of Starlette, so the backend can be exercised over real HTTP with uvicorn.

It is not used at runtime: requirements.txt installs the real FastAPI.
"""
import inspect
import json
from contextlib import asynccontextmanager

from pydantic import BaseModel, ValidationError
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

__all__ = ["FastAPI", "HTTPException"]
SHIM = True


class HTTPException(Exception):
    def __init__(self, status_code: int, detail=None):
        self.status_code, self.detail = status_code, detail


def _convert(value, annotation):
    if annotation in (int, "int"):
        return int(value)
    if annotation in (float, "float"):
        return float(value)
    if annotation in (bool, "bool"):
        return value.lower() in ("1", "true", "yes", "on")
    return value


class FastAPI:
    def __init__(self, title=None, **kwargs):
        self.title = title
        self._routes = []
        self._mounts = []
        self._startup = []
        self._middleware = []
        self._asgi = None
        self.routes = []   # (method, path, name) for inspection in tests

    # -- registration -----------------------------------------------------
    def add_middleware(self, cls, **options):
        self._middleware.append((cls, options))

    def on_event(self, event):
        def deco(fn):
            if event == "startup":
                self._startup.append(fn)
            return fn
        return deco

    def _route(self, method, path):
        def deco(fn):
            self._routes.append((method, path, fn))
            self.routes.append((method, path, fn.__name__))
            return fn
        return deco

    def get(self, path, **kw):
        return self._route("GET", path)

    def post(self, path, **kw):
        return self._route("POST", path)

    def mount(self, path, app, name=None):
        self._mounts.append(Mount(path, app=app, name=name))

    # -- ASGI -------------------------------------------------------------
    def _endpoint(self, fn):
        sig = inspect.signature(fn)

        async def endpoint(request: Request):
            kwargs = {}
            try:
                for name, p in sig.parameters.items():
                    ann = p.annotation
                    if inspect.isclass(ann) and issubclass(ann, BaseModel):
                        kwargs[name] = ann.model_validate(await request.json())
                    elif name in request.path_params:
                        kwargs[name] = _convert(request.path_params[name], ann)
                    elif name in request.query_params:
                        kwargs[name] = _convert(request.query_params[name], ann)
                    elif p.default is inspect.Parameter.empty:
                        return JSONResponse({"detail": f"missing parameter {name}"}, status_code=422)
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                return JSONResponse({"detail": str(exc)}, status_code=422)
            try:
                result = await run_in_threadpool(fn, **kwargs)
            except HTTPException as exc:
                return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
            # strict JSON encoding (no numpy fallback) so that non-serialisable
            # return values are caught by the tests just as FastAPI would fail on them
            return Response(json.dumps(result, allow_nan=False), media_type="application/json")
        return endpoint

    def _build(self):
        routes = [Route(path, self._endpoint(fn), methods=[m]) for m, path, fn in self._routes] + self._mounts

        @asynccontextmanager
        async def lifespan(app):
            for fn in self._startup:
                await run_in_threadpool(fn)
            yield

        app = Starlette(routes=routes, lifespan=lifespan)
        for cls, options in self._middleware:
            app.add_middleware(cls, **options)
        return app

    async def __call__(self, scope, receive, send):
        if self._asgi is None:
            self._asgi = self._build()
        await self._asgi(scope, receive, send)
