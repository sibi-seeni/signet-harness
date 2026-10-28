"""Loading a corpus (docs.json) and a question set (questions.json). The formats
are documented in experiments/template/README.md."""

import hashlib
import json
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class Passage:
    pid: str
    doc_id: str
    family: str
    filename: str
    title: str
    text: str
    locator: str
    ingest: dict


@dataclass
class Question:
    id: str
    case: str
    question: str
    family: str = ""
    corpus: list = field(default_factory=list)          # doc ids "in the library" for this question
    gold_passages: list = field(default_factory=list)   # always retrieved when force_gold is on
    gold_answer: str = ""
    alt_answer: str = ""
    accept: list = field(default_factory=list)          # regexes for the gold answer
    alt: list = field(default_factory=list)             # regexes for the alternative (stale / attacked) answer
    pilot: bool = False
    live: bool = False
    verified: bool = False
    extra: dict = field(default_factory=dict)           # any other field in the file, kept as-is

    @classmethod
    def from_dict(cls, d: dict) -> "Question":
        known = {f.name for f in fields(cls)} - {"extra"}
        return cls(**{k: v for k, v in d.items() if k in known},
                   extra={**d.get("extra", {}), **{k: v for k, v in d.items() if k not in known | {"extra"}}})


class Corpus:
    def __init__(self, docs_path: Path):
        raw = json.loads(Path(docs_path).read_text())
        self.docs = {d["doc_id"]: d for d in raw["docs"]}
        self.passages: dict[str, Passage] = {}
        for d in raw["docs"]:
            for p in d["passages"]:
                self.passages[p["pid"]] = Passage(
                    pid=p["pid"], doc_id=d["doc_id"], family=d.get("family", ""),
                    filename=d.get("filename", d["doc_id"]), title=d.get("title", d["doc_id"]), text=p["text"],
                    locator=p.get("locator", ""), ingest=d.get("ingest", {}),
                )

    def pool_for(self, q: Question) -> list[Passage]:
        """Passages 'in the corpus' for this question: its own docs plus the
        default docs of every other family as distractors. A question with an
        empty corpus list sees every document."""
        if not q.corpus:
            return list(self.passages.values())
        doc_ids = set(q.corpus)
        for d in self.docs.values():
            if d.get("default_pool") and d.get("family", "") != q.family:
                doc_ids.add(d["doc_id"])
        return [p for p in self.passages.values() if p.doc_id in doc_ids]

    def get(self, pids: list[str]) -> list[Passage]:
        return [self.passages[p] for p in pids]


def load_questions(path: Path) -> list[Question]:
    raw = json.loads(Path(path).read_text())
    return [Question.from_dict(q) for q in raw["questions"]]


def check_questions(questions: list[Question], corpus: Corpus | None = None) -> list[str]:
    """Problems with a question set: duplicate ids, missing core fields, unknown docs
    or gold passages, regexes that don't compile. Empty list = fine."""
    import re
    problems, seen = [], set()
    for q in questions:
        if q.id in seen:
            problems.append(f"{q.id}: duplicate id")
        seen.add(q.id)
        if not q.case or not q.question:
            problems.append(f"{q.id}: missing case or question")
        for pattern in q.accept + q.alt:
            try:
                re.compile(pattern)
            except re.error as e:
                problems.append(f"{q.id}: bad regex {pattern!r}: {e}")
        if corpus is not None:
            problems += [f"{q.id}: unknown doc {d}" for d in q.corpus if d not in corpus.docs]
            problems += [f"{q.id}: unknown gold passage {p}" for p in q.gold_passages if p not in corpus.passages]
    return problems


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
