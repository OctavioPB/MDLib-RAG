"""Concept maps (ADR-022): the LLM reads one fragment at a time; code checks, merges and lays out.

1. `extract`: one LLM call per fragment returns concepts and relations as JSON. A concept is kept
   only if it is written in its fragment (accents and case ignored, whole words), a relation only if
   both ends were kept. An invalid reply is asked once more, then the fragment is skipped.
2. `build_map`: concepts are merged across fragments by name and by embedding similarity, the
   hierarchy is chosen by how connected each concept is, and nodes get radial coordinates.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field

import numpy as np

from biblio.config import MapsConfig
from biblio.generate.ask import load_prompt
from biblio.generate.context import Fragment
from biblio.generate.llm import LLMClient, Message

Embed = Callable[[Sequence[str]], "np.ndarray"]

_MAX_CONCEPT_CHARS = 60
_MAX_CONCEPT_WORDS = 6
_MAX_LABEL_CHARS = 40
# words the prompt forbids as concepts; a model that ignores it is corrected here
_GENERIC = {
    "ejemplo",
    "ejemplos",
    "modelo",
    "modelos",
    "dato",
    "datos",
    "seccion",
    "secciones",
    "concepto",
    "conceptos",
    "guia",
    "guias",
    "tema",
    "fragmento",
    "texto",
    "nota",
    "notas",
    "metodo",
    "sistema",
}


def fold(text: str) -> str:
    """Lower case, no accents, words separated by single spaces: what "the same name" means."""
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().casefold()
    return " ".join(re.findall(r"\w+", plain))


# Common words of each language, folded: enough to tell an English note from a Spanish one (ADR-042)
_SPANISH_WORDS = frozenset(
    {"el", "la", "los", "las", "de", "del", "que", "y", "en", "un", "una", "por", "para", "con"}
    | {"es", "se", "su", "al", "lo", "como", "mas", "pero", "sus", "este", "esta", "son"}
)
_ENGLISH_WORDS = frozenset(
    {"the", "of", "and", "to", "in", "is", "that", "for", "it", "with", "as", "was", "on", "be"}
    | {"by", "this", "are", "from", "or", "an", "which", "have", "not", "can"}
)


def in_english(text: str) -> bool:
    """Whether a note's text is in English rather than Spanish: a 4B model told only in general
    to copy names as written still translated a whole English section (ADR-042)."""
    words = fold(text).split()
    return sum(w in _ENGLISH_WORDS for w in words) > sum(w in _SPANISH_WORDS for w in words)


# --- 1. extraction ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Extraction:
    fragment: int  # Fragment.number
    concepts: tuple[str, ...]  # as written by the model, checked against the fragment
    relations: tuple[tuple[str, str, str], ...]  # (from, to, label), both ends in `concepts`
    # ADR-042: a note in another language keeps its own names; the model adds a Spanish rendering
    translations: tuple[tuple[str, str], ...] = ()  # (concept as written, its Spanish name)
    topic_names: tuple[str, ...] = ()  # how the fragment itself names the topic («burnout»)


def _json_object(text: str) -> dict[str, object] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def parse_extraction(reply: str, fragment: Fragment, max_concepts: int) -> Extraction | None:
    """The model's JSON, checked against the fragment text; None if the reply is unusable."""
    data = _json_object(reply)
    raw_concepts = data.get("conceptos") if data is not None else None
    if data is None or not isinstance(raw_concepts, list):
        return None
    haystack = f" {fold(fragment.text)} "
    kept: dict[str, str] = {}  # folded → as written
    spanish: dict[str, str] = {}  # folded → the model's Spanish name, when it differs (ADR-042)
    for item in raw_concepts:
        # {"nombre": as written, "es": in Spanish}, or just the name
        raw: object = item.get("nombre") if isinstance(item, dict) else item
        translated: object = item.get("es") if isinstance(item, dict) else None
        if not isinstance(raw, str):
            continue
        name = " ".join(raw.split()).strip(" .,;:«»\"'")
        key = fold(name)
        if (
            not key
            or len(name) > _MAX_CONCEPT_CHARS
            or len(key.split()) > _MAX_CONCEPT_WORDS
            or key in _GENERIC
            or f" {key} " not in haystack  # not written in the fragment: dropped (ADR-022)
            or key in kept
        ):
            continue
        kept[key] = name
        if isinstance(translated, str):
            es = " ".join(translated.split()).strip(" .,;:«»\"'")
            if fold(es) and fold(es) != key and len(es) <= _MAX_CONCEPT_CHARS:
                spanish[key] = es
        if len(kept) == max_concepts:
            break
    relations: list[tuple[str, str, str]] = []
    raw_relations = data.get("relaciones", [])
    for rel in raw_relations if isinstance(raw_relations, list) else []:
        if not isinstance(rel, dict):
            continue
        a, b, label = rel.get("de"), rel.get("a"), rel.get("tipo")
        if not (isinstance(a, str) and isinstance(b, str) and isinstance(label, str)):
            continue
        ka, kb = fold(a), fold(b)
        label = " ".join(label.split()).strip(" .").lower()
        if ka in kept and kb in kept and ka != kb and 0 < len(label) <= _MAX_LABEL_CHARS:
            triple = (kept[ka], kept[kb], label)
            if triple not in relations:
                relations.append(triple)
    raw_topic = data.get("tema") if data is not None else None
    named = (
        [raw_topic]
        if isinstance(raw_topic, str)
        else raw_topic
        if isinstance(raw_topic, list)
        else []
    )
    topic_names = tuple(
        dict.fromkeys(
            " ".join(n.split())
            for n in named
            if isinstance(n, str) and fold(n) and f" {fold(n)} " in haystack  # written in it
        )
    )
    return Extraction(
        fragment.number,
        tuple(kept.values()),
        tuple(relations),
        tuple((kept[k], es) for k, es in spanish.items()),
        topic_names,
    )


def extract(llm: LLMClient, cfg: MapsConfig, topic: str, fragment: Fragment) -> Extraction | None:
    system = load_prompt("maps_system")
    user = load_prompt("maps_user").format(
        topic=topic,
        citation=fragment.citation,
        section=fragment.text[: cfg.max_fragment_chars],
        max_concepts=cfg.concepts_per_fragment,
    )
    if in_english(fragment.text):
        user += "\n\n" + load_prompt("maps_english").format()
    messages = (Message("system", system), Message("user", user))
    for _ in range(2):  # one retry, then the fragment is skipped
        reply = "".join(llm.stream(messages, cfg.extract_max_tokens))
        extraction = parse_extraction(reply, fragment, cfg.concepts_per_fragment)
        if extraction is not None:
            return extraction
    return None


# --- 2. merging, hierarchy and layout -----------------------------------------------------------


@dataclass(frozen=True)
class MapNode:
    id: str
    label: str
    level: int  # 0 the topic, 1 around it, 2 under a level-1 concept
    parent: str | None
    sources: tuple[int, ...]  # fragment numbers
    x: int
    y: int
    # the Spanish name of a concept a note names in another language (ADR-042)
    translation: str = ""

    @property
    def shown(self) -> str:
        """The name with its translation, for exports: «replacement cost (costo de reemplazo)»."""
        return f"{self.label} ({self.translation})" if self.translation else self.label


@dataclass(frozen=True)
class MapEdge:
    source: str
    target: str
    label: str
    kind: str  # "tree" (drawn as the hierarchy) | "cross" (a relation across branches)


@dataclass(frozen=True)
class ConceptMap:
    topic: str
    nodes: tuple[MapNode, ...]
    edges: tuple[MapEdge, ...]
    omitted: int  # concepts left out: weak, with no clear parent, or over the caps ([maps])


@dataclass
class _Concept:
    labels: Counter[str]
    folds: set[str]
    fragments: set[int]
    translations: Counter[str] = field(default_factory=Counter)  # Spanish names (ADR-042)

    @property
    def label(self) -> str:
        top = max(self.labels.values())
        return min((name for name, n in self.labels.items() if n == top), key=len)

    @property
    def translation(self) -> str:
        """The Spanish name most often given, unless the label already is that name."""
        for name, _ in self.translations.most_common():
            if fold(name) != fold(self.label):
                return name
        return ""


def merge_concepts(
    extractions: Sequence[Extraction], embed: Embed | None, threshold: float
) -> list[_Concept]:
    """Same folded name → one concept; then names whose embeddings are close enough → one."""
    groups: dict[str, _Concept] = {}
    for ex in extractions:
        spanish = dict(ex.translations)
        for name in ex.concepts:
            key = fold(name)
            group = groups.setdefault(key, _Concept(Counter(), {key}, set()))
            group.labels[name] += 1
            group.fragments.add(ex.fragment)
            if name in spanish:
                group.translations[spanish[name]] += 1
    concepts = list(groups.values())
    if len(concepts) < 2:
        return concepts
    parent = list(range(len(concepts)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # a note's concept whose Spanish name another note writes is that concept (ADR-042)
    index = {key: i for i, key in enumerate(groups)}
    for i, c in enumerate(concepts):
        for name in c.translations:
            j = index.get(fold(name))
            if j is not None and j != i:
                parent[root(j)] = root(i)
    if embed is not None:
        vectors = np.asarray(embed([c.label for c in concepts]), dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        similar = vectors @ vectors.T
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                if similar[i, j] >= threshold:
                    parent[root(j)] = root(i)
    merged: dict[int, _Concept] = {}
    for i, c in enumerate(concepts):
        target = merged.setdefault(root(i), _Concept(Counter(), set(), set()))
        target.labels.update(c.labels)
        target.folds |= c.folds
        target.fragments |= c.fragments
        target.translations.update(c.translations)
    return list(merged.values())


_ARTICLES = {"el", "la", "los", "las", "lo", "un", "una", "unos", "unas"}


def _core(folded: str) -> str:
    """Without leading articles and a plural ending: «las químicas» → «quimica»."""
    words = folded.split()
    while len(words) > 1 and words[0] in _ARTICLES:
        words = words[1:]
    text = " ".join(words)
    for ending in ("es", "s"):
        if text.endswith(ending) and len(text) > len(ending) + 3:
            return text[: -len(ending)]
    return text


def _is_topic(folded: str, topic: str) -> bool:
    """The topic itself, give or take an article or a plural, becomes the centre. A name that only
    contains the topic is a concept of its own: «química orgánica» and «bioquímica» are branches of
    «química», not the same thing (merging them emptied the map of its main concepts)."""
    return folded == topic or _core(folded) == _core(topic)


def topic_in_text(topic: str, text: str) -> bool:
    """Whether the topic is written in the text, give or take accents, case, an article or a plural
    (a gap when no fragment of a map has it, ADR-030)."""
    core = _core(fold(topic))
    return bool(core) and re.search(rf"(?<!\w){re.escape(core)}", fold(text)) is not None


def build_map(
    topic: str, extractions: Sequence[Extraction], embed: Embed | None, cfg: MapsConfig
) -> ConceptMap:
    concepts = merge_concepts(extractions, embed, cfg.merge_similarity)
    topic_key = fold(topic)
    # how notes in other languages name the topic: «burnout» for «agotamiento laboral» (ADR-042)
    aliases = {fold(n) for ex in extractions for n in ex.topic_names} - {""}
    ids: dict[str, str] = {}  # folded name → node id
    root_fragments: set[int] = set()
    others: dict[str, _Concept] = {}
    for i, c in enumerate(concepts):
        if any(_is_topic(k, topic_key) or k in aliases for k in c.folds):
            root_fragments |= c.fragments
            for k in c.folds:
                ids[k] = "root"
        else:
            node_id = f"c{i}"
            others[node_id] = c
            for k in c.folds:
                ids[k] = node_id

    relations: list[tuple[str, str, str]] = []
    for ex in extractions:
        for a, b, label in ex.relations:
            ia, ib = ids.get(fold(a)), ids.get(fold(b))
            if ia and ib and ia != ib and (ia, ib, label) not in relations:
                relations.append((ia, ib, label))
    degree: Counter[str] = Counter()
    linked: dict[tuple[str, str], str] = {}  # unordered pair → first label seen, in its direction
    for a, b, label in relations:
        degree[a] += 1
        degree[b] += 1
        linked.setdefault((a, b), label)
    near_root = {b if a == "root" else a for a, b, _ in relations if "root" in (a, b)}

    def score(node_id: str) -> tuple[float, str]:
        """Breadth first: a concept found in several fragments matters more than one with many
        relations in a single list (a method's five steps once outranked the topic's branches)."""
        c = others[node_id]
        value = 3 * len(c.fragments) + min(degree[node_id], 3) + (3 if node_id in near_root else 0)
        return (-value, c.label)

    def weak(node_id: str) -> bool:
        """One word, one fragment, no relation: usually a verb or a generic noun ("predecir")."""
        c = others[node_id]
        return len(c.label.split()) == 1 and len(c.fragments) == 1 and degree[node_id] == 0

    omitted = sum(1 for n in others if weak(n))
    ranked = sorted((n for n in others if not weak(n)), key=score)
    level1 = ranked[: cfg.level1_max]
    children: dict[str, list[str]] = {n: [] for n in level1}
    budget = cfg.max_nodes - 1 - len(level1)
    for node_id in ranked[cfg.level1_max :]:
        c = others[node_id]

        def weight(p: str, node_id: str = node_id, c: _Concept = c) -> int:
            """A child needs a relation to its parent stated in a fragment (ADR-022): fragments
            in common are not enough, since branches listed together («química inorgánica»,
            «química física») ended up nested under each other. Shared fragments break ties."""
            related = ((p, node_id) in linked) + ((node_id, p) in linked)
            shared = len(c.fragments & others[p].fragments)
            return 3 * related + shared if related else 0

        options = [p for p in level1 if len(children[p]) < cfg.children_max and weight(p) > 0]
        if budget <= 0 or not options:
            omitted += 1
            continue
        best = max(options, key=lambda p: (weight(p), -level1.index(p)))
        children[best].append(node_id)
        budget -= 1

    positions = _radial_layout(level1, children)
    nodes = [MapNode("root", topic, 0, None, tuple(sorted(root_fragments)), 0, 0)]
    for p in level1:
        x, y = positions[p]
        c = others[p]
        nodes.append(
            MapNode(p, c.label, 1, "root", tuple(sorted(c.fragments)), x, y, c.translation)
        )
        for child in children[p]:
            x, y = positions[child]
            c = others[child]
            nodes.append(
                MapNode(child, c.label, 2, p, tuple(sorted(c.fragments)), x, y, c.translation)
            )

    def tree_edge(parent: str, child: str) -> MapEdge:
        if (parent, child) in linked:
            return MapEdge(parent, child, linked[(parent, child)], "tree")
        if (child, parent) in linked:
            return MapEdge(child, parent, linked[(child, parent)], "tree")
        return MapEdge(parent, child, "", "tree")

    edges = [tree_edge(n.parent, n.id) for n in nodes if n.parent]
    kept = {n.id for n in nodes}
    tree_pairs = {frozenset((e.source, e.target)) for e in edges}
    for (a, b), label in linked.items():
        pair = frozenset((a, b))
        if a in kept and b in kept and pair not in tree_pairs:
            edges.append(MapEdge(a, b, label, "cross"))
            tree_pairs.add(pair)
    return ConceptMap(topic, tuple(nodes), tuple(edges), omitted)


def _radial_layout(
    level1: Sequence[str], children: dict[str, list[str]]
) -> dict[str, tuple[int, int]]:
    """Level-1 concepts on a circle around the topic, each with a sector sized by its children;
    children on an outer ring inside their parent's sector (alternating radii when crowded)."""
    if not level1:
        return {}
    weights = [1 + len(children[p]) for p in level1]
    total = sum(weights)
    r1 = 260 + 30 * max(0, len(level1) - 6)
    r2 = r1 + 240
    positions: dict[str, tuple[int, int]] = {}
    angle = -math.pi / 2 - math.pi * weights[0] / total  # the first concept straight above
    for p, w in zip(level1, weights, strict=True):
        sector = 2 * math.pi * w / total
        mid = angle + sector / 2
        positions[p] = (round(r1 * math.cos(mid)), round(r1 * math.sin(mid)))
        kids = children[p]
        for j, child in enumerate(kids):
            a = angle + sector * (j + 0.5) / len(kids)
            r = r2 + (70 if len(kids) > 2 and j % 2 else 0)
            positions[child] = (round(r * math.cos(a)), round(r * math.sin(a)))
        angle += sector
    return positions


def map_outline(cmap: ConceptMap) -> list[str]:
    """The map as indented lines: `- concept [1][3]` under its parent (CLI and Markdown export)."""
    lines = []
    by_parent: dict[str | None, list[MapNode]] = defaultdict(list)
    for n in cmap.nodes:
        by_parent[n.parent].append(n)
    names = {n.id: n.label for n in cmap.nodes}
    tree = {frozenset((e.source, e.target)): e for e in cmap.edges if e.kind == "tree"}

    def cites(n: MapNode) -> str:
        return "".join(f"[{s}]" for s in n.sources)

    def relation(parent: str, child: str) -> str:
        """The stated relation as a sentence, so its direction is clear."""
        e = tree.get(frozenset((parent, child)))
        if e is None or not e.label:
            return ""
        return f" — _{names[e.source]} {e.label} {names[e.target]}_"

    for n1 in by_parent["root"]:
        lines.append(f"- **{n1.shown}** {cites(n1)}{relation('root', n1.id)}")
        for n2 in by_parent[n1.id]:
            lines.append(f"  - {n2.shown} {cites(n2)}{relation(n1.id, n2.id)}")
    return lines


# --- 3. the whole run ---------------------------------------------------------------------------


class CachedEmbed:
    """Embeds concept names once: the map is rebuilt after every fragment with mostly the same
    names, and the query embedder runs on CPU."""

    def __init__(self, embed: Embed) -> None:
        self._embed = embed
        self._cache: dict[str, np.ndarray] = {}

    def __call__(self, names: Sequence[str]) -> np.ndarray:
        missing = [n for n in dict.fromkeys(names) if n not in self._cache]
        if missing:
            for name, vector in zip(missing, self._embed(missing), strict=True):
                self._cache[name] = np.asarray(vector, dtype=np.float32)
        return np.stack([self._cache[n] for n in names])


@dataclass(frozen=True)
class MapProgress:
    processed: int  # fragments read so far
    skipped: int  # fragments whose reply was unusable twice
    map: ConceptMap
    # the notes name the topic in their own words (an English note's «burnout», ADR-042)
    topic_named: bool = False


def generate_map(
    llm: LLMClient,
    cfg: MapsConfig,
    topic: str,
    fragments: Sequence[Fragment],
    embed: Embed | None,
    llm_turn: Callable[[], AbstractContextManager[object]] = nullcontext,
) -> Iterator[MapProgress]:
    """Read the fragments one by one and yield the map as it grows (ADR-022)."""
    extractions: list[Extraction] = []
    skipped = 0
    for i, fragment in enumerate(fragments, start=1):
        with llm_turn():
            extraction = extract(llm, cfg, topic, fragment)
        if extraction is None:
            skipped += 1
        else:
            extractions.append(extraction)
        # only English notes count: a Spanish one must write the topic itself (ADR-029, 030)
        english = {f.number for f in fragments if in_english(f.text)}
        named = any(ex.topic_names and ex.fragment in english for ex in extractions)
        yield MapProgress(i, skipped, build_map(topic, extractions, embed, cfg), named)
