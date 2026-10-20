"""Flowcharts, sequence and cause-effect diagrams in Mapas (ADR-036)."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from biblio.config import Config
from biblio.generate.ask import load_prompt
from biblio.generate.context import Fragment
from biblio.generate.llm import Message
from biblio.index.indexer import index_library
from biblio.maps.build import ConceptMap, MapEdge, MapNode
from biblio.maps.diagrams import (
    END,
    Cause,
    Fishbone,
    Flow,
    effect_label,
    flow_edges,
    flow_versions,
    frames,
    kind_from_spanish,
    merge_causes,
    move_cause,
    move_step,
    parse_causes,
    parse_groups,
    parse_sequence,
    parse_steps,
    parse_suggestion,
)
from biblio.maps.drawing import (
    diagram_from_page,
    diagram_page,
    fishbone_mermaid,
    flow_mermaid,
    layout_fishbone,
    layout_flow,
    layout_sequence,
    novak_layout,
    novak_mermaid,
    sequence_mermaid,
    wrap,
)
from biblio.retrieve.rerank import RerankPlan
from biblio.retrieve.store import open_index
from biblio.web.engine import Engine
from biblio.web.server import make_server

from fakes import FakeEmbedder, WordCounter

PROCESS = (
    "El descenso de gradiente empieza por inicializar los parámetros al azar. "
    "Después se calcula el gradiente de la función de pérdida. "
    "Luego se actualizan los parámetros restando el gradiente por la tasa de aprendizaje. "
    "Si el algoritmo no converge, se vuelve a calcular el gradiente; si converge, termina."
)
RATE = (
    "Antes de entrenar hay que elegir la tasa de aprendizaje con cuidado: una tasa grande "
    "hace que el descenso de gradiente oscile."
)
CALLS = (
    "Step Functions envía una tarea a Lambda con un token. Lambda guarda el token en SQS. "
    "Cuando el trabajo termina, el servicio externo llama a SendTaskSuccess con el token. "
    "Si la tarea falla, el servicio llama a SendTaskFailure."
)
CAUSES = (
    "Un modelo se degrada en producción por la deriva de datos, cuando los datos nuevos "
    "dejan de parecerse a los de entrenamiento. También influyen las etiquetas con ruido y "
    "un preprocesado distinto en producción."
)


def fragment(n: int, text: str) -> Fragment:
    return Fragment(n, f"g{n}.md", ("Guía", "Sección"), text, 1, 5)


STEPS = {
    "pasos": [
        {
            "n": 1,
            "paso": "Inicializar los parámetros",
            "cita": "inicializar los parámetros al azar",
        },
        {"n": 2, "paso": "Calcular el gradiente", "cita": "se calcula el gradiente de la función"},
        {
            "n": 3,
            "paso": "Actualizar los parámetros",
            "cita": "se actualizan los parámetros restando el gradiente",
        },
        {
            "n": 4,
            "paso": "Converge",
            "tipo": "decision",
            "si": "fin",
            "no": 2,
            "cita": "Si el algoritmo no converge",
        },
        {"n": 5, "paso": "Usar una red neuronal", "cita": "se entrena una red neuronal"},
    ]
}


# --- flowcharts ---------------------------------------------------------------------------------


def test_steps_need_a_quote_from_the_fragment_and_decisions_a_way_out() -> None:
    steps = parse_steps(json.dumps(STEPS), fragment(1, PROCESS), 12)
    assert steps is not None
    assert [s.label for s in steps] == [
        "Inicializar los parámetros",
        "Calcular el gradiente",
        "Actualizar los parámetros",
        "¿Converge?",  # a decision is written as a question
    ]  # the step without a quote in the text is dropped
    decision = steps[3]
    assert decision.decision and decision.yes == END and decision.no == steps[1].id
    assert [s.position for s in steps] == sorted(s.position for s in steps)
    lost = {
        "pasos": [
            {
                "n": 1,
                "paso": "¿Sigue?",
                "tipo": "decision",
                "si": 9,
                "cita": "Si el algoritmo no converge",
            }
        ]
    }
    only = parse_steps(json.dumps(lost), fragment(1, PROCESS), 12)
    assert only is not None and not only[0].decision  # a question going nowhere is a step
    both = {"pasos": [{**lost["pasos"][0], "si": "fin", "no": "fin"}]}
    same = parse_steps(json.dumps(both), fragment(1, PROCESS), 12)
    assert same is not None and not same[0].decision  # both answers to the end: not a fork
    assert parse_steps('{"pasos": []}', fragment(1, PROCESS), 12) == []
    assert parse_steps("no es JSON", fragment(1, PROCESS), 12) is None


def test_arrows_are_cited_only_in_the_order_of_the_text() -> None:
    steps = parse_steps(json.dumps(STEPS), fragment(1, PROCESS), 12)
    assert steps is not None
    flow = Flow(1, tuple(steps))
    edges = {(e.source, e.target): e for e in flow_edges(flow)}
    s = [x.id for x in steps]
    assert not edges[("start", s[0])].inferred
    assert not edges[(s[0], s[1])].inferred and not edges[(s[2], s[3])].inferred
    assert edges[(s[3], END)].label == "sí" and edges[(s[3], s[1])].label == "no"  # the loop
    swapped = move_step(flow, s[1], 1)  # «Actualizar» before «Calcular»: not the text's order
    assert swapped is not None
    edges = {(e.source, e.target): e for e in flow_edges(swapped)}
    assert edges[(s[2], s[1])].inferred and not edges[(s[0], s[2])].inferred
    assert move_step(flow, s[0], -1) is None and move_step(flow, "x", 1) is None


def test_each_note_is_a_version_of_the_process() -> None:
    main = parse_steps(json.dumps(STEPS), fragment(1, PROCESS), 12)
    other = parse_steps(
        json.dumps(
            {
                "pasos": [
                    {
                        "n": 1,
                        "paso": "Elegir la tasa de aprendizaje",
                        "cita": "elegir la tasa de aprendizaje",
                    },
                    {
                        "n": 2,
                        "paso": "Calcular el gradiente",
                        "cita": "el descenso de gradiente oscile",
                    },
                ]
            }
        ),
        fragment(2, RATE),
        12,
    )
    assert main is not None and other is not None
    versions = flow_versions([other, main, []])
    assert [v.source for v in versions] == [1, 2]  # most steps first; no empty versions
    assert [s.label for s in versions[0].steps] == [s.label for s in main]  # never mixed
    calc = next(s for s in versions[0].steps if s.label == "Calcular el gradiente")
    assert calc.sources == (1, 2)  # written in the other note too
    assert versions[1].steps[1].sources == (2, 1)
    assert [v.source for v in flow_versions([other, main], 3)] == [1]  # two steps: no process


def test_flow_layout_mermaid_and_loops() -> None:
    steps = parse_steps(json.dumps(STEPS), fragment(1, PROCESS), 12)
    assert steps is not None
    flow = Flow(1, tuple(steps))
    drawing = layout_flow(flow)
    roles = [b.role for b in drawing.boxes]
    assert roles[0] == "start" and roles[-1] == "end" and roles.count("decision") == 1
    assert all(b.x >= 0 and b.x + b.w <= drawing.width for b in drawing.boxes)
    loop = next(ln for ln in drawing.lines if ln.label == "no")
    step_left = min(b.x for b in drawing.boxes if b.role == "step")
    assert len(loop.points) == 4 and loop.points[1][0] < step_left  # back up, on the left
    text = flow_mermaid(flow)
    assert text.startswith("flowchart TD") and '{"¿Converge?"}' in text
    assert (
        f'{steps[3].id} -->|"no"| {steps[1].id}' in text and f'{steps[3].id} -->|"sí"| fin' in text
    )
    assert wrap("una dos tres cuatro cinco seis", 9, 2) == ("una dos", "tres cua…")


# --- sequence diagrams --------------------------------------------------------------------------


def test_participants_must_be_named_and_messages_quoted() -> None:
    reply = {
        "participantes": ["Step Functions", "Lambda", "servicio externo", "DynamoDB"],
        "mensajes": [
            {
                "de": "Step Functions",
                "a": "Lambda",
                "mensaje": "Enviar tarea con token",
                "cita": "envía una tarea a Lambda con un token",
            },
            {
                "de": "lambda",
                "a": "Lambda",
                "mensaje": "Guardar el token",
                "cita": "Lambda guarda el token en SQS",
            },
            {
                "de": "servicio externo",
                "a": "Step Functions",
                "mensaje": "SendTaskSuccess",
                "respuesta": True,
                "cita": "llama a SendTaskSuccess con el token",
                "condicion": "si termina",
            },
            {
                "de": "servicio externo",
                "a": "Step Functions",
                "mensaje": "SendTaskFailure",
                "respuesta": True,
                "cita": "el servicio llama a SendTaskFailure",
                "condicion": "si la tarea falla",
            },
            {
                "de": "DynamoDB",
                "a": "Lambda",
                "mensaje": "Leer",
                "cita": "DynamoDB devuelve el registro",
            },
        ],
    }
    version = parse_sequence(json.dumps(reply), fragment(1, CALLS), 14, 6)
    assert version is not None
    assert version.participants == (
        "Step Functions",
        "Lambda",
        "servicio externo",
    )  # DynamoDB: not in the text
    assert [m.label for m in version.messages][1] == "Guardar el token"  # a message to itself
    assert len(version.messages) == 4 and version.messages[2].reply
    assert frames(version.messages) == [(2, 2, "si termina"), (3, 3, "si la tarea falla")]
    text = sequence_mermaid(version)
    assert "participant p0 as Step Functions" in text
    assert "  alt si termina\n  p2-->>p0: SendTaskSuccess\n  else si la tarea falla" in text
    assert text.rstrip().endswith("end")
    drawing = layout_sequence(version)
    assert len([ln for ln in drawing.lines if ln.role == "lifeline"]) == 3
    assert len(drawing.frames) == 2 and drawing.lines[-1].dashed  # a reply is dashed
    assert parse_sequence('{"participantes": [], "mensajes": []}', fragment(1, CALLS), 14, 6) == (
        replace(version, participants=(), messages=())
    )


# --- cause-effect -------------------------------------------------------------------------------


def test_causes_are_quoted_merged_and_grouped() -> None:
    reply = {
        "causas": [
            {"causa": "deriva de datos", "cita": "por la deriva de datos"},
            {"causa": "Etiquetas con ruido", "cita": "las etiquetas con ruido"},
            {"causa": "Falta de GPU", "cita": "no hay GPU suficiente"},
        ]
    }
    causes = parse_causes(json.dumps(reply), fragment(1, CAUSES), 8)
    assert causes is not None and [c.label for c in causes] == [
        "Deriva de datos",
        "Etiquetas con ruido",
    ]
    again = [
        Cause("f2c1", "La deriva de datos", "q", (2,)),
        Cause("f2c2", "Preprocesado distinto", "q", (2,)),
    ]
    merged, over = merge_causes(causes, again, 3)
    assert [c.sources for c in merged] == [(1, 2), (1,), (2,)] and over == 0
    merged, over = merge_causes(merged, [Cause("f3c1", "Umbral fijo de decisión", "q", (3,))], 3)
    assert over == 1  # over the cap
    groups = parse_groups(
        json.dumps(
            {
                "categorias": [
                    {"nombre": "datos", "causas": [1, 2]},
                    {"nombre": "Despliegue", "causas": [2]},
                ]
            }
        ),
        3,
        None,
        6,
    )
    assert groups == [("Datos", [0, 1]), ("Otras", [2])]  # a cause twice keeps its first place
    six = ["Método", "Máquina", "Materiales", "Mano de obra", "Medición", "Medio ambiente"]
    groups = parse_groups(
        json.dumps(
            {
                "categorias": [
                    {"nombre": "medicion", "causas": [3]},
                    {"nombre": "Datos", "causas": [1]},
                    {"nombre": "Método", "causas": [2]},
                ]
            }
        ),
        3,
        six,
        6,
    )
    assert groups == [("Método", [1]), ("Medición", [2]), ("Otras", [0])]  # fixed names and order
    assert parse_groups('{"categorias": []}', 3, None, 6) is None
    assert (
        effect_label("¿Por qué se degrada un modelo en producción?")
        == "Se degrada un modelo en producción"
    )
    assert effect_label("causas del sobreajuste") == "Sobreajuste"


def test_fishbone_layout_moves_and_mindmap() -> None:
    causes = tuple(Cause(f"c{i}", f"Causa número {i}", "q", (1,)) for i in range(5))
    fishbone = Fishbone(
        "El modelo se degrada",
        causes,
        (("Datos", ("c0", "c1")), ("Modelo", ("c2",)), ("Despliegue (prod)", ("c3", "c4"))),
        "model",
        0,
    )
    drawing = layout_fishbone(fishbone)
    cats = [b for b in drawing.boxes if b.role == "category"]
    spine = next(ln for ln in drawing.lines if ln.role == "spine")
    assert len(cats) == 3 and cats[0].y < spine.points[0][1] < cats[1].y  # above and below
    effect = next(b for b in drawing.boxes if b.role == "effect")
    assert spine.points[-1][0] == effect.x and spine.arrow
    moved = move_cause(fishbone, "c0", "Modelo")
    assert moved is not None and moved.categories[1] == ("Modelo", ("c2", "c0"))
    assert move_cause(fishbone, "c0", "Nada") is None
    text = fishbone_mermaid(fishbone)
    assert text.startswith("mindmap\n  root((El modelo se degrada))") and "Despliegue prod" in text


# --- the Novak map and the type suggestion -------------------------------------------------------


def test_novak_layout_is_top_down_with_linking_phrases() -> None:
    nodes = (
        MapNode("root", "regresión", 0, None, (1,), 0, 0),
        MapNode("c1", "regresión lineal", 1, "root", (1,), 300, 10),
        MapNode("c2", "mínimos cuadrados", 2, "c1", (1,), 500, 40),
        MapNode("c3", "regresión logística", 1, "root", (1,), -300, 10),
    )
    edges = (
        MapEdge("c1", "root", "es un tipo de", "tree"),
        MapEdge("c1", "c2", "se ajusta con", "tree"),
        MapEdge("c3", "c2", "no usa", "cross"),
    )
    laid = novak_layout(ConceptMap("regresión", nodes, edges, 0))
    x = {n.id: n.x for n in laid.nodes}
    y = {n.id: n.y for n in laid.nodes}
    assert x["root"] < x["c1"] == x["c3"] < x["c2"]  # left to right: no line crosses another
    assert y["c1"] == y["c2"] and y["c1"] < y["root"] < y["c3"]  # the topic between its branches
    text = novak_mermaid(laid)
    assert 'c1 -->|"es un tipo de"| root' in text and 'c3 -.->|"no usa"| c2' in text
    assert parse_suggestion('{"tipo": "causa_efecto", "motivo": "Es una falla."}') is not None
    assert kind_from_spanish("Mapa conceptual") == "novak" and kind_from_spanish("otro") is None
    assert parse_suggestion('{"tipo": "tarta"}') is None


# --- the web API --------------------------------------------------------------------------------

GUIDES = {
    "ml/descenso.md": "# Descenso\n\n## Algoritmo\n\n" + PROCESS,
    "ml/tasa.md": "# Tasa\n\n## Elegir la tasa\n\n" + RATE,
    "aws/tokens.md": "# Tokens\n\n## Callback\n\n" + CALLS,
    "mlops/fallas.md": "# Fallas\n\n## Degradación\n\n" + CAUSES,
}


class _DiagramLLM:
    """Answers by prompt: the extraction prompts by fragment, the short ones by kind."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def stream(self, messages: Sequence[Message], max_tokens: int) -> Iterator[str]:
        system, user = messages[0].content, messages[1].content
        name = next(
            n
            for n in (
                "flow_system",
                "sequence_system",
                "cause_system",
                "cause_group_system",
                "diagram_suggest_system",
                "maps_system",
            )
            if system == load_prompt(n)
        )
        self.calls.append(name)
        if name == "maps_system":
            concepts = ["descenso de gradiente", "tasa de aprendizaje"]
            relation = {
                "de": "tasa de aprendizaje",
                "a": "descenso de gradiente",
                "tipo": "controla",
            }
            yield json.dumps({"conceptos": concepts, "relaciones": [relation]})
            return
        if name == "flow_system":
            if "inicializar los parámetros" in user:
                yield json.dumps(STEPS)
            elif "elegir la tasa" in user:
                rate = [
                    ("Preparar el entrenamiento", "Antes de entrenar hay que"),
                    ("Elegir la tasa de aprendizaje", "elegir la tasa de aprendizaje"),
                    ("Vigilar las oscilaciones", "el descenso de gradiente oscile"),
                ]
                steps = [{"n": i, "paso": s, "cita": q} for i, (s, q) in enumerate(rate, 1)]
                yield json.dumps({"pasos": steps})
            else:
                yield '{"pasos": []}'
        elif name == "sequence_system":
            if "SendTaskSuccess" in user:
                yield json.dumps(
                    {
                        "participantes": ["Step Functions", "Lambda"],
                        "mensajes": [
                            {
                                "de": "Step Functions",
                                "a": "Lambda",
                                "mensaje": "Enviar tarea",
                                "cita": "envía una tarea a Lambda",
                            }
                        ],
                    }
                )
            else:
                yield '{"participantes": [], "mensajes": []}'
        elif name == "cause_system":
            if "deriva de datos" in user and "Efecto: Nada" not in user:
                yield json.dumps(
                    {
                        "causas": [
                            {"causa": "Deriva de datos", "cita": "por la deriva de datos"},
                            {"causa": "Etiquetas con ruido", "cita": "las etiquetas con ruido"},
                        ]
                    }
                )
            else:
                yield '{"causas": []}'
        elif name == "cause_group_system":
            if "Método" in user:
                yield json.dumps({"categorias": [{"nombre": "Materiales", "causas": [1, 2]}]})
            else:
                yield json.dumps({"categorias": [{"nombre": "Datos", "causas": [1, 2]}]})
        else:
            yield json.dumps({"tipo": "flujo", "motivo": "Describe pasos en orden."})

    def close(self) -> None:
        pass


@pytest.fixture
def api(config: Config, tmp_path: Path) -> Iterator[tuple[tuple[str, int], _DiagramLLM]]:
    library = tmp_path / "library"
    for rel, text in GUIDES.items():
        (library / rel).parent.mkdir(parents=True, exist_ok=True)
        (library / rel).write_text(text, encoding="utf-8")
    cfg = replace(
        config,
        chunking=replace(config.chunking, min_tokens=5, target_tokens=60, max_tokens=80),
        paths=replace(
            config.paths,
            library=library,
            index_db=tmp_path / "index.db",
            drafts_dir=tmp_path / "drafts",
        ),
        maps=replace(config.maps, history_db=tmp_path / "maps.db"),
    )
    index_library(
        cfg,
        library,
        full=True,
        dtype="float32",
        embedder_factory=FakeEmbedder,
        counter_factory=WordCounter,
    )
    llm = _DiagramLLM()
    engine = Engine(
        cfg,
        open_index(cfg, shared_across_threads=True),
        FakeEmbedder(),
        plan_resolver=lambda _c: RerankPlan(None, 0, "test"),
        llm_factory=lambda _c: llm,
        counter_factory=lambda _c: WordCounter(),
    )
    engine.llm_running = lambda: True  # type: ignore[method-assign]
    srv = make_server(engine, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield ("127.0.0.1", srv.server_address[1]), llm
    srv.shutdown()
    srv.server_close()
    engine.close()


def _post(server: tuple[str, int], path: str, body: Any) -> tuple[int, Any]:
    import http.client

    conn = http.client.HTTPConnection(*server, timeout=10)
    conn.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    raw = resp.read().decode("utf-8")
    conn.close()
    if resp.getheader("Content-Type", "").startswith("text/event-stream"):
        events = []
        for block in raw.strip().split("\n\n"):
            lines = dict(line.split(": ", 1) for line in block.splitlines())
            events.append((lines["event"], json.loads(lines["data"])))
        return resp.status, events
    return resp.status, json.loads(raw)


def _get(server: tuple[str, int], path: str) -> tuple[int, Any]:
    import http.client

    conn = http.client.HTTPConnection(*server, timeout=10)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = json.loads(resp.read().decode("utf-8"))
    conn.close()
    return resp.status, body


def _final(events: list[tuple[str, Any]]) -> tuple[dict[str, Any], Any]:
    pages = [d for n, d in events if n == "diagram"]
    done = next(d for n, d in events if n == "done")
    return pages[-1]["map"], done


def test_a_flow_is_streamed_edited_and_saved(api: Any, tmp_path: Path) -> None:
    server, llm = api
    status, events = _post(server, "/api/maps", {"topic": "descenso de gradiente", "kind": "flow"})
    assert status == 200 and events[0][0] == "start"
    page, done = _final(events)
    stages = [d["stage"] for n, d in events if n == "diagram"]
    assert stages[-1] == "done" and llm.calls.count("flow_system") == len(events[0][1]["fragments"])
    versions = page["data"]["versions"]
    assert page["data"]["current"] == 0 and len(versions) == 2
    labels = [s["label"] for s in versions[0]["steps"]]
    assert labels[0] == "Inicializar los parámetros" and len(labels) == 4
    assert not any(ln["dashed"] for ln in page["drawing"]["lines"])  # all in the text's order
    assert page["mermaid"].startswith("flowchart TD")
    map_id = done["map_id"]
    _status, history = _get(server, "/api/maps/history")
    assert history["maps"][0]["kind"] == "flow" and history["maps"][0]["concepts"] == len(labels)
    step = versions[0]["steps"][1]["id"]
    status, edited = _post(
        server, "/api/maps/edit", {"id": map_id, "op": "move_step", "step": step, "delta": -1}
    )
    assert status == 200 and edited["map"]["data"]["versions"][0]["steps"][0]["id"] == step
    assert any(ln["dashed"] for ln in edited["map"]["drawing"]["lines"])  # out of the text order
    _status, stored = _get(server, f"/api/maps/history/{map_id}")
    assert stored["map"]["data"]["versions"][0]["steps"][0]["id"] == step  # kept
    status, other = _post(server, "/api/maps/edit", {"id": map_id, "op": "version", "index": 1})
    assert status == 200 and other["map"]["data"]["current"] == 1 and other["map"]["count"] == 3
    _post(server, "/api/maps/edit", {"id": map_id, "op": "version", "index": 0})
    assert (
        _post(
            server, "/api/maps/edit", {"id": map_id, "op": "move_step", "step": step, "delta": -1}
        )[0]
        == 404
    )
    assert _post(server, "/api/maps/edit", {"id": map_id, "op": "borrar"})[0] == 400
    status, saved = _post(server, "/api/maps/save", {"id": map_id})
    assert status == 200 and saved["markdown"].endswith("-flujo.md")
    markdown = Path(saved["markdown"]).read_text(encoding="utf-8")
    assert (
        "```mermaid\nflowchart TD" in markdown
        and "«inicializar los parámetros al azar»" in markdown
    )
    canvas = json.loads(Path(saved["canvas"]).read_text(encoding="utf-8"))
    assert {"start", "end"} <= {n["id"] for n in canvas["nodes"]} and canvas["edges"]
    assert _get(server, f"/api/maps/history/{map_id}/mastery")[1]["nodes"] == {}


def test_a_sequence_and_a_fishbone_regrouped(api: Any) -> None:
    server, _llm = api
    _status, events = _post(
        server, "/api/maps", {"topic": "callback con token", "kind": "sequence"}
    )
    page, done = _final(events)
    assert page["data"]["versions"][0]["participants"] == ["Step Functions", "Lambda"]
    assert page["mermaid"].startswith("sequenceDiagram") and done["map_id"]
    assert (
        _post(server, "/api/maps/edit", {"id": done["map_id"], "op": "version", "index": 3})[0]
        == 404
    )

    _status, events = _post(
        server, "/api/maps", {"topic": "por qué se degrada un modelo", "kind": "fishbone"}
    )
    page, done = _final(events)
    assert page["data"]["effect"] == "Se degrada un modelo"
    assert page["data"]["categories"] == [{"name": "Datos", "causes": ["f1c1", "f1c2"]}]
    assert page["data"]["grouping"] == "model"
    status, regrouped = _post(server, "/api/maps/regroup", {"id": done["map_id"], "grouping": "6m"})
    assert status == 200 and regrouped["map"]["data"]["categories"][0]["name"] == "Materiales"
    status, _custom = _post(
        server,
        "/api/maps/regroup",
        {"id": done["map_id"], "grouping": "custom", "categories": ["Solo una"]},
    )
    assert status == 400
    cause = page["data"]["causes"][0]["id"]
    status, _moved = _post(
        server,
        "/api/maps/edit",
        {"id": done["map_id"], "op": "move_cause", "cause": cause, "category": "Otra"},
    )
    assert status == 404  # no such category
    kind, fishbone, _ = diagram_from_page(regrouped["map"])
    assert kind == "fishbone" and isinstance(fishbone, Fishbone)
    assert diagram_page("fishbone", "t", fishbone)["count"] == 2


def test_nothing_of_that_kind_is_not_stored_and_types_are_suggested(api: Any) -> None:
    server, _llm = api
    _status, events = _post(server, "/api/maps", {"topic": "nada que ver", "kind": "fishbone"})
    _page, done = _final(events)
    assert done["map_id"] is None  # no causes in the notes: nothing kept
    assert _post(server, "/api/maps", {"topic": "x", "kind": "tarta"})[0] == 400
    status, suggestion = _post(server, "/api/maps/suggest", {"topic": "descenso de gradiente"})
    assert status == 200 and suggestion == {"kind": "flow", "reason": "Describe pasos en orden."}
    status, events = _post(server, "/api/maps", {"topic": "descenso", "kind": "novak"})
    final = [d for n, d in events if n == "map"][-1]["map"]
    assert final["kind"] == "novak"
    _status, history = _get(server, "/api/maps/history")
    assert history["maps"][0]["kind"] == "novak"
