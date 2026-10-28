"""The freshness check from Track 2, as a library: arm A4, "the harness always
checks". Before the model answers, code looks every retrieved document up in a
version registry and writes the result next to the passage ("SUPERSEDED by Rev 5").
It met the Track 2 closure rule on both models (stale 44% -> 0%, no false flags).

    registry = MockRegistry.from_file("experiments/track2_freshness/data/registry.json")
    annotations = annotate(passages, registry)              # pid -> "NIST-SP-800-53 Rev 4: WITHDRAWN, ..."
    statuses = assess(passages, registry)                   # doc_id -> {"status", "latest", ...} for a flag
    answer = with_unverified_warning(answer, annotations)   # if the registry was down, say so in code

Known limits (CLOSURE_REPORT.md): it can't see documents amended in place under
the same version label (C4, 0/12 for both models; `assess` reports the in-place
changes the registry knows about so a caller can flag them), and documents with
no date or contradictory dates can only be flagged, not fixed.

The annotation text is exactly what Track 2's A4 arm sent; tests/test_parity.py
checks it against the logged requests of every Track 2 run.
"""

import json
import re
from pathlib import Path

# What A4 adds to the system prompt so the model knows where the annotations come from.
ANNOTATION_NOTE = ("Each passage carries a 'Registry check' line written by the system after looking the "
                   "document up in the authoritative version registry.")

UNAVAILABLE_ANNOTATION = "REGISTRY UNAVAILABLE (timeout). Freshness not verified."

# Closure recommendation 3: when the registry is down, the harness adds this itself.
# (In Track 2, Gemma passed the annotation on in only 4 of 16 runs.)
UNVERIFIED_WARNING = ("Note: the freshness of the sources behind this answer could not be verified "
                      "because the version registry was unavailable.")

CHECK_LATEST_VERSION_DECL = {
    "name": "check_latest_version",
    "description": ("Look up a document in the authoritative version registry. Returns the version, "
                    "effective date, status (current / superseded / withdrawn), what supersedes it, "
                    "the latest version, and any in-place changes on record."),
    "parameters": {
        "type": "object",
        "properties": {
            "title_or_id": {"type": "string",
                            "description": "Document title, filename or identifier, including any version or date you know."},
        },
        "required": ["title_or_id"],
    },
}


def _norm(s: str) -> str:
    return re.sub(r"[_/]+", " ", s.lower())


def _has_version_alias(text: str, alias: str) -> bool:
    return re.search(r"(?<![\w.])" + re.escape(alias) + r"(?!\w)", text) is not None


class MockRegistry:
    """A frozen version registry (see experiments/track2_freshness/data/registry.json for
    the format). `unavailable=True` makes every lookup time out."""

    def __init__(self, registry: dict, unavailable: bool = False, timeout_s: int = 10):
        self.families = registry["families"]
        self.unavailable = unavailable
        self.timeout_s = timeout_s

    @classmethod
    def from_file(cls, path, **kw) -> "MockRegistry":
        return cls(json.loads(Path(path).read_text()), **kw)

    def lookup(self, title_or_id: str) -> dict:
        if self.unavailable:
            return {"error": f"TIMEOUT: version registry did not respond within {self.timeout_s}s",
                    "query": title_or_id}
        q = _norm(title_or_id or "")
        hits = [name for name, fam in self.families.items()
                if any(a in q for a in fam["aliases"])]
        if not hits:
            return {"query": title_or_id, "found": False,
                    "message": "No registry entry matches this title or id."}
        results = [self._family_result(name, q) for name in hits]
        out = results[0] if len(results) == 1 else {"results": results}
        out["query"] = title_or_id
        return out

    def _family_result(self, name: str, q: str) -> dict:
        fam = self.families[name]
        matched = [v for v in fam["versions"]
                   if any(_has_version_alias(q, a) for a in v["aliases"])]
        if not matched:
            return {
                "found": True, "family": name, "version": "unknown", "latest": fam["latest"],
                "note": "The query doesn't identify a version, so the document's own version can't be confirmed. Known versions:",
                "versions": [self._entry(name, fam, v) for v in fam["versions"]],
            }
        entries = [self._entry(name, fam, v) for v in matched]
        if len(entries) == 1:
            return {"found": True, **entries[0]}
        return {"found": True, "family": name, "latest": fam["latest"],
                "note": "Several versions match the query.", "versions": entries}

    @staticmethod
    def _entry(name: str, fam: dict, v: dict) -> dict:
        e = {"family": name, "version": v["version"], "effective": v["effective"],
             "status": v["status"], "superseded_by": v.get("superseded_by"), "latest": fam["latest"]}
        for k in ("withdrawn", "changes"):
            if k in v:
                e[k] = v[k]
        return e


def summarize_lookup(result: dict) -> str:
    """One-line annotation for a registry lookup result."""
    if "error" in result:
        return UNAVAILABLE_ANNOTATION
    if not result.get("found", True):
        return "No registry entry found for this document."
    if "results" in result:
        return " / ".join(summarize_lookup(r) for r in result["results"])
    if result.get("version") == "unknown":
        known = ", ".join(v["version"] for v in result.get("versions", []))
        return (f"Version of this document could not be identified (family {result['family']}; "
                f"known versions: {known}; latest: {result['latest']}).")
    if "versions" in result:
        return " / ".join(_entry_line(v) for v in result["versions"])
    return _entry_line(result)


def _entry_line(e: dict) -> str:
    line = f"{e['family']} {e['version']}: {e['status'].upper()}"
    if e.get("superseded_by"):
        line += f", SUPERSEDED by {e['superseded_by']}"
    if e.get("withdrawn"):
        line += f" (withdrawn {e['withdrawn']})"
    line += f"; latest: {e['latest']}"
    for c in e.get("changes", []):
        label = f"Change {c['change']}" if "change" in c else f"release {c.get('release')}"
        line += f"; in-place {label} dated {c['date']}"
    return line + "."


def query_for(passage) -> str:
    """What the harness looks up for a passage: its document title and filename."""
    return f"{passage.title} ({passage.filename})"


def status_of(result: dict) -> dict:
    """A registry lookup as a machine-readable status for a flag:
    status is current / superseded / withdrawn / unverifiable / ambiguous / not_found / unable_to_verify."""
    if "error" in result:
        return {"status": "unable_to_verify", "latest": None, "in_place_changes": []}
    if not result.get("found", True):
        return {"status": "not_found", "latest": None, "in_place_changes": []}
    if "results" in result or ("versions" in result and result.get("version") != "unknown"):
        return {"status": "ambiguous", "latest": result.get("latest"), "in_place_changes": []}
    if result.get("version") == "unknown":
        return {"status": "unverifiable", "latest": result["latest"], "in_place_changes": []}
    return {"status": result["status"], "latest": result["latest"], "version": result["version"],
            "superseded_by": result.get("superseded_by"), "in_place_changes": result.get("changes", [])}


def assess(passages, registry: MockRegistry, log=None) -> dict:
    """Look up each distinct document once. Returns doc_id -> {"status", "latest", ...,
    "annotation", "registry"}. `log(name, args, result)` is called for every lookup,
    e.g. a RunContext's log_tool, so lookups land in the trace."""
    out = {}
    for p in passages:
        if p.doc_id in out:
            continue
        query = query_for(p)
        result = registry.lookup(query)
        if log:
            log("check_latest_version", {"title_or_id": query}, result)
        out[p.doc_id] = {**status_of(result), "annotation": summarize_lookup(result), "registry": result}
    return out


def annotate(passages, registry: MockRegistry, log=None) -> dict:
    """pid -> the 'Registry check' line Track 2's A4 put on each passage."""
    by_doc = assess(passages, registry, log)
    return {p.pid: by_doc[p.doc_id]["annotation"] for p in passages}


def unverified(annotations: dict) -> bool:
    """Did any lookup fail because the registry was unavailable?"""
    return any(a == UNAVAILABLE_ANNOTATION for a in annotations.values())


def with_unverified_warning(answer: str, annotations: dict) -> str:
    """Append UNVERIFIED_WARNING to the answer, in code, if the registry was down."""
    if not unverified(annotations) or UNVERIFIED_WARNING in (answer or ""):
        return answer
    return f"{answer.rstrip()}\n\n{UNVERIFIED_WARNING}" if answer else UNVERIFIED_WARNING
