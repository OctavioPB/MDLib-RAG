"""HTTP server for the web UI (ADR-016): standard library only, loopback only.

Defenses, because any page open in the browser can try to reach `127.0.0.1`:
- binds only to a loopback address;
- rejects requests whose `Host` header is not the address it serves (DNS rebinding);
- API calls must be `POST` with a JSON body, and no CORS headers are sent (no cross-site reads);
- a strict Content-Security-Policy: scripts, styles and data only from this server.
"""

from __future__ import annotations

import ipaddress
import json
from collections.abc import Callable, Generator
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any

from biblio.generate.draft import OutlineSection
from biblio.practice.quiz import SCOPE_KINDS
from biblio.practice.store import KINDS, RESULTS
from biblio.web.drafts import list_drafts, read_draft
from biblio.web.engine import BusyError, Engine, Event, hit_json, outline_json

_STATIC = resources.files("biblio.web") / "static"
_STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/static/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/static/app.css": ("app.css", "text/css; charset=utf-8"),
    "/static/guias.js": ("guias.js", "text/javascript; charset=utf-8"),
    "/static/practicar.js": ("practicar.js", "text/javascript; charset=utf-8"),
    "/static/theme.js": ("theme.js", "text/javascript; charset=utf-8"),
}
_MAX_BODY = 64 * 1024
_MAX_QUERY = 2000
_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


_MAX_TITLE = 200
_MAX_NOTES = 1000
_MAX_SECTIONS = 20


def parse_sections(raw: Any) -> list[OutlineSection] | None:
    """The outline as edited in the page: [{title, notes?}], or None if it is not valid."""
    if not isinstance(raw, list) or not 1 <= len(raw) <= _MAX_SECTIONS:
        return None
    sections = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        title, notes = item.get("title"), item.get("notes", "")
        if not isinstance(title, str) or not isinstance(notes, str):
            return None
        title = " ".join(title.split())  # one line: it becomes a `## ` heading
        if not title or len(title) > _MAX_TITLE or len(notes) > _MAX_NOTES:
            return None
        sections.append(OutlineSection(title, " ".join(notes.split())))
    return sections


class LoopbackOnlyError(ValueError):
    """The web UI may only listen on a loopback address."""


def require_loopback_host(host: str) -> None:
    if host == "localhost":
        return
    try:
        if ipaddress.ip_address(host).is_loopback:
            return
    except ValueError:
        pass
    raise LoopbackOnlyError(
        f"Refusing to listen on {host!r}: the web UI is loopback-only (ADR-016)."
    )


def make_handler(engine: Engine, allowed_hosts: set[str]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "biblio"
        sys_version = ""

        def log_message(self, format: str, *args: Any) -> None:
            pass  # queries are private; do not echo them to the console

        # --- plumbing ---------------------------------------------------------------------------

        def _host_allowed(self) -> bool:
            if self.headers.get("Host", "") in allowed_hosts:
                return True
            self._send_json({"error": "forbidden host"}, HTTPStatus.FORBIDDEN)
            return False

        def _headers(self, status: HTTPStatus, content_type: str, length: int | None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            if length is not None:
                self.send_header("Content-Length", str(length))
            for name, value in _SECURITY_HEADERS.items():
                self.send_header(name, value)
            self.end_headers()

        def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._headers(status, "application/json; charset=utf-8", len(body))
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any] | None:
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self._send_json({"error": "JSON body required"}, HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
                return None
            length = int(self.headers.get("Content-Length") or 0)
            if length > _MAX_BODY:
                self._send_json({"error": "body too large"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                return None
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:  # malformed JSON or a body that is not valid UTF-8
                payload = None
            if not isinstance(payload, dict):
                self._send_json({"error": "invalid JSON"}, HTTPStatus.BAD_REQUEST)
                return None
            return payload

        def _text_field(
            self, payload: dict[str, Any], name: str, max_length: int = _MAX_QUERY
        ) -> str | None:
            value = payload.get(name)
            if not isinstance(value, str) or not value.strip() or len(value) > max_length:
                self._send_json(
                    {"error": f"'{name}' must be a non-empty string"}, HTTPStatus.BAD_REQUEST
                )
                return None
            return value.strip()

        def _scope_field(self, payload: dict[str, Any]) -> tuple[bool, str | None]:
            """(ok, folder): a missing or empty scope means the whole library."""
            scope = payload.get("scope")
            if scope in (None, ""):
                return True, None
            if isinstance(scope, str) and scope.strip("/") in dict(engine.folders()):
                return True, scope.strip("/")
            self._send_json({"error": "unknown folder in 'scope'"}, HTTPStatus.BAD_REQUEST)
            return False, None

        def _choice_field(
            self, payload: dict[str, Any], name: str, allowed: tuple[str, ...]
        ) -> str | None:
            """The value of `name`, or its first allowed value when missing; None after a 400."""
            value = payload.get(name, allowed[0])
            if value in allowed:
                return str(value)
            self._send_json(
                {"error": f"'{name}' must be one of {', '.join(allowed)}"}, HTTPStatus.BAD_REQUEST
            )
            return None

        # --- routes -----------------------------------------------------------------------------

        def do_GET(self) -> None:
            if not self._host_allowed():
                return
            if self.path == "/api/status":
                self._send_json(engine.status())
                return
            if self.path == "/api/folders":
                folders = [{"path": p, "documents": n} for p, n in engine.folders()]
                self._send_json({"folders": folders})
                return
            drafts_dir = engine.config.paths.drafts_dir
            if self.path == "/api/practice/guides":
                guides = [{"path": p, "title": t} for p, t in engine.practice_guides()]
                self._send_json({"guides": guides})
                return
            if self.path == "/api/practice/progress":
                self._send_json(engine.practice_progress())
                return
            if self.path == "/api/drafts":
                self._send_json({"drafts": [asdict(d) for d in list_drafts(drafts_dir)]})
                return
            if self.path.startswith("/api/drafts/"):
                detail = read_draft(drafts_dir, self.path.removeprefix("/api/drafts/"))
                if detail is None:
                    self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                    return
                self._send_json(
                    {
                        "slug": detail.slug,
                        "topic": detail.topic,
                        "outline": outline_json(detail.outline) if detail.outline else None,
                        "markdown": detail.markdown,
                    }
                )
                return
            static = _STATIC_FILES.get(self.path)
            if static is None:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            name, content_type = static
            body = (_STATIC / name).read_bytes()
            self._headers(HTTPStatus.OK, content_type, len(body))
            self.wfile.write(body)

        def do_POST(self) -> None:
            if not self._host_allowed():
                return
            routes = {
                "/api/search": self._search,
                "/api/ask": self._ask,
                "/api/drafts/outline": self._draft_outline,
                "/api/drafts/write": self._draft_write,
                "/api/practice/start": self._practice_start,
                "/api/practice/check": self._practice_check,
                "/api/practice/reveal": self._practice_reveal,
                "/api/practice/self": self._practice_self,
                "/api/practice/grade": self._practice_grade,
                "/api/practice/review": self._practice_review,
                "/api/practice/flag": self._practice_flag,
            }
            route = routes.get(self.path)
            if route is None:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            payload = self._read_json()
            if payload is None:
                return
            route(payload)

        def _search(self, payload: dict[str, Any]) -> None:
            query = self._text_field(payload, "query")
            if query is None:
                return
            k = payload.get("k", engine.config.retrieval.final_k)
            if not isinstance(k, int) or not 1 <= k <= 50:
                k = engine.config.retrieval.final_k
            precision = self._choice_field(payload, "precision", ("precise", "fast"))
            ok, scope = self._scope_field(payload)
            if precision is None or not ok:
                return
            result = engine.search(query, k, precise=precision == "precise", scope=scope)
            self._send_json(
                {
                    "retriever": result.retriever,
                    "milliseconds": round(result.milliseconds),
                    "hits": [hit_json(h, engine) for h in result.hits],
                }
            )

        def _ask(self, payload: dict[str, Any]) -> None:
            question = self._text_field(payload, "question")
            if question is None:
                return
            strategy = self._choice_field(payload, "strategy", ("direct", "map-reduce"))
            ok, scope = self._scope_field(payload)
            if strategy is None or not ok:
                return
            map_reduce = "on" if strategy == "map-reduce" else "off"  # opt-in only (ADR-017)
            self._stream(lambda: engine.ask(question, scope=scope, map_reduce=map_reduce))

        def _draft_outline(self, payload: dict[str, Any]) -> None:
            topic = self._text_field(payload, "topic", _MAX_TITLE)
            if topic is not None:
                self._stream(lambda: engine.propose_outline(topic))

        def _draft_write(self, payload: dict[str, Any]) -> None:
            topic = self._text_field(payload, "topic", _MAX_TITLE)
            if topic is None:
                return
            sections = parse_sections(payload.get("sections"))
            if sections is None:
                self._send_json(
                    {"error": f"'sections' must be 1 to {_MAX_SECTIONS} items with a title"},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            self._stream(lambda: engine.write_draft(topic, sections))

        # --- practice (ADR-020) -----------------------------------------------------------------

        def _id_field(self, payload: dict[str, Any], name: str) -> int | None:
            value = payload.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
            self._send_json(
                {"error": f"'{name}' must be a positive integer"}, HTTPStatus.BAD_REQUEST
            )
            return None

        def _not_found(self) -> None:
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def _practice_start(self, payload: dict[str, Any]) -> None:
            scope = payload.get("scope")
            kind = scope.get("kind") if isinstance(scope, dict) else None
            value = scope.get("value", "") if isinstance(scope, dict) else None
            count = payload.get("count")
            kinds = payload.get("kinds")
            limit = engine.config.practice.max_questions
            valid = (
                kind in SCOPE_KINDS
                and isinstance(value, str)
                and len(value) <= _MAX_QUERY
                and (kind == "review" or value.strip())
                and isinstance(count, int)
                and not isinstance(count, bool)
                and 1 <= count <= limit
                and isinstance(kinds, list)
                and kinds
                and all(k in KINDS for k in kinds)
            )
            if not valid:
                self._send_json(
                    {
                        "error": f"expected scope {{kind: {'|'.join(SCOPE_KINDS)}, value}}, "
                        f"count 1 to {limit} and kinds from {', '.join(KINDS)}"
                    },
                    HTTPStatus.BAD_REQUEST,
                )
                return
            assert isinstance(value, str) and isinstance(count, int) and isinstance(kinds, list)
            if kind == "folder" and value.strip("/") not in dict(engine.folders()):
                self._send_json({"error": "unknown folder"}, HTTPStatus.BAD_REQUEST)
                return
            self._stream(lambda: engine.start_quiz(str(kind), value.strip(), count, kinds))

        def _practice_check(self, payload: dict[str, Any]) -> None:
            qid = self._id_field(payload, "question_id")
            if qid is None:
                return
            choice = payload.get("choice")
            if not isinstance(choice, bool | int):
                self._send_json(
                    {"error": "'choice' must be an index or a boolean"}, HTTPStatus.BAD_REQUEST
                )
                return
            result = engine.check_choice(qid, choice)
            self._send_json(result) if result is not None else self._not_found()

        def _practice_reveal(self, payload: dict[str, Any]) -> None:
            qid = self._id_field(payload, "question_id")
            if qid is None:
                return
            result = engine.reveal(qid)
            self._send_json(result) if result is not None else self._not_found()

        def _practice_self(self, payload: dict[str, Any]) -> None:
            qid = self._id_field(payload, "question_id")
            answer = payload.get("answer", "")
            result = payload.get("result")
            if qid is None:
                return
            if not isinstance(answer, str) or len(answer) > _MAX_QUERY or result not in RESULTS:
                self._send_json(
                    {"error": f"'result' must be one of {', '.join(RESULTS)}"},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            attempt = engine.self_grade(qid, answer, str(result))
            self._send_json({"attempt_id": attempt}) if attempt else self._not_found()

        def _practice_grade(self, payload: dict[str, Any]) -> None:
            qid = self._id_field(payload, "question_id")
            if qid is None:
                return
            answer = self._text_field(payload, "answer")
            if answer is None:
                return
            if not engine.llm_running():
                self._send_json(
                    {"error": "No LLM server running. Start it with scripts/serve-llm.ps1."},
                    HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return
            result = engine.model_grade(qid, answer)
            self._send_json(result) if result is not None else self._not_found()

        def _practice_review(self, payload: dict[str, Any]) -> None:
            attempt = self._id_field(payload, "attempt_id")
            result = payload.get("result")
            if attempt is None:
                return
            if result not in RESULTS:
                self._send_json(
                    {"error": f"'result' must be one of {', '.join(RESULTS)}"},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            ok = engine.practice.review_model_grade(attempt, str(result))
            self._send_json({"ok": True}) if ok else self._not_found()

        def _practice_flag(self, payload: dict[str, Any]) -> None:
            qid = self._id_field(payload, "question_id")
            note = payload.get("note", "")
            if qid is None:
                return
            if not isinstance(note, str) or len(note) > _MAX_NOTES:
                self._send_json({"error": "'note' must be a short string"}, HTTPStatus.BAD_REQUEST)
                return
            ok = engine.practice.flag(qid, note.strip())
            self._send_json({"ok": True}) if ok else self._not_found()

        def _stream(self, start: Callable[[], Generator[Event, None, None]]) -> None:
            """Server-sent events from an engine generator. The first event is taken before the
            headers go out, so "no LLM" (503) and "busy" (409) are plain JSON errors."""
            if not engine.llm_running():
                self._send_json(
                    {"error": "No LLM server running. Start it with scripts/serve-llm.ps1."},
                    HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return
            events = start()
            try:
                first = next(events, None)
            except BusyError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.CONFLICT)
                return
            except ValueError as exc:  # nothing to work on, e.g. a scope with no content sections
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._headers(HTTPStatus.OK, "text/event-stream; charset=utf-8", None)
            try:
                if first is not None:
                    self._event(*first)
                for event, data in events:
                    self._event(event, data)
            except (BrokenPipeError, ConnectionResetError):
                events.close()  # the page went away: stop generating (see engine._streamed)
            except Exception as exc:  # report to the page instead of dropping the stream
                self._event("error", {"error": str(exc)})

        def _event(self, event: str, data: dict[str, Any]) -> None:
            chunk = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            self.wfile.write(chunk.encode("utf-8"))
            self.wfile.flush()

    return Handler


def make_server(engine: Engine, host: str, port: int) -> ThreadingHTTPServer:
    require_loopback_host(host)
    server = ThreadingHTTPServer((host, port), make_handler(engine, set()))
    server.daemon_threads = True
    bound = server.server_address[1]
    hosts = {f"{host}:{bound}"}
    if host in ("127.0.0.1", "localhost"):
        hosts |= {f"127.0.0.1:{bound}", f"localhost:{bound}"}
    server.RequestHandlerClass = make_handler(engine, hosts)
    return server
