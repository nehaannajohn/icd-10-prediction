"""ICD-10-CM FY2025 Tabular List as a tree: lookup, hierarchy levels, ancestors, validation, notes."""
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from src.common.config import load_config

LEVELS = ["chapter", "block", "category", "subcategory_4", "subcategory_5", "subcategory_6", "full_code"]
_DIAG_LEVELS = LEVELS[2:]          # <diag> nesting depth 1..5 -> category .. full_code
NOTE_KINDS = ["includes", "inclusionTerm", "excludes1", "excludes2", "codeFirst",
              "useAdditionalCode", "codeAlso", "notes", "sevenChrNote"]


@dataclass
class Node:
    code: str                      # "Ch4", "E08-E13", "E11", "E11.29"
    description: str
    level: str
    parent: str | None
    children: list[str] = field(default_factory=list)
    notes: dict[str, list[str]] = field(default_factory=dict)
    seventh: dict[str, str] = field(default_factory=dict)   # own sevenChrDef: char -> meaning


@dataclass
class Validation:
    code: str
    valid: bool
    reason: str                    # ok | not_found | not_billable | seventh_char_required | invalid_seventh_char


def normalize(code: str) -> str:
    """'e1129 ' -> 'E11.29'. Chapter/block ids are returned unchanged."""
    code = code.strip().upper()
    if code.startswith("CH") or "-" in code:
        return code.replace("CH", "Ch", 1)
    code = code.replace(".", "")
    return code if len(code) <= 3 else f"{code[:3]}.{code[3:]}"


class ICD10Tree:
    def __init__(self, nodes: dict[str, Node]):
        self.nodes = nodes

    @property
    def chapters(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.level == "chapter"]

    def get(self, code: str) -> Node | None:
        return self.nodes.get(normalize(code))

    def children(self, code: str) -> list[Node]:
        node = self.get(code)
        return [self.nodes[c] for c in node.children] if node else []

    def path(self, code: str) -> list[Node]:
        """Chapter -> ... -> node, for a code that exists in the tree."""
        node, out = self.get(code), []
        while node:
            out.append(node)
            node = self.nodes.get(node.parent) if node.parent else None
        return out[::-1]

    def resolve(self, code: str) -> Node | None:
        """Deepest tree code that is a prefix of `code` (handles 7th-character codes and invalid codes)."""
        key = normalize(code)
        while key:
            if key in self.nodes:
                return self.nodes[key]
            key = key[:-1].rstrip(".")
        return None

    def seventh_chars(self, code: str) -> dict[str, str]:
        """7th-character options inherited from the nearest ancestor (or self) that defines them."""
        for node in reversed(self.path(code)):
            if node.seventh:
                return node.seventh
        return {}

    def notes(self, code: str) -> dict[str, list[dict]]:
        """Own + inherited notes, chapter first: {kind: [{"from": code, "text": ...}]}."""
        out: dict[str, list[dict]] = {}
        for node in self.path(code):
            for kind, texts in node.notes.items():
                out.setdefault(kind, []).extend({"from": node.code, "text": t} for t in texts)
        return out

    def validate(self, code: str) -> Validation:
        c = normalize(code)
        node = self.nodes.get(c)
        if node and node.level not in ("chapter", "block"):
            if node.children:
                return Validation(c, False, "not_billable")
            if len(c.replace(".", "")) < 7 and self.seventh_chars(c):
                return Validation(c, False, "seventh_char_required")
            return Validation(c, True, "ok")
        stem = self.resolve(c)
        raw, stem_raw = c.replace(".", ""), (stem.code.replace(".", "") if stem else "")
        if (stem and stem.level not in ("chapter", "block") and not stem.children and len(raw) == 7
                and set(raw[len(stem_raw):-1]) <= {"X"}):
            ok = raw[-1] in self.seventh_chars(stem.code)
            return Validation(c, ok, "ok" if ok else "invalid_seventh_char")
        return Validation(c, False, "not_found")

    def ancestor(self, code: str, level: str) -> str:
        """Code's ancestor at `level`; a code shallower than `level` maps to its deepest tree node.
        `full_code` is always the normalized code itself. Unknown codes map to themselves."""
        c = normalize(code)
        if level == "full_code":
            return c
        stem = self.resolve(c)
        if stem is None:
            return c
        target = LEVELS.index(level)
        return [n for n in self.path(stem.code) if LEVELS.index(n.level) <= target][-1].code


def _notes(el: ET.Element) -> dict[str, list[str]]:
    out = {}
    for kind in NOTE_KINDS:
        texts = ["".join(n.itertext()).strip() for sub in el.findall(kind) for n in sub.findall("note")]
        if texts:
            out[kind] = texts
    return out


def _add(nodes: dict[str, Node], node: Node) -> None:
    if node.code in nodes:
        raise ValueError(f"duplicate id in tabular: {node.code}")
    nodes[node.code] = node
    if node.parent:
        nodes[node.parent].children.append(node.code)


def _add_diag(nodes: dict[str, Node], el: ET.Element, parent: str, depth: int) -> None:
    code = el.findtext("name")
    _add(nodes, Node(code, el.findtext("desc"), _DIAG_LEVELS[depth], parent, notes=_notes(el),
                     seventh={e.get("char"): "".join(e.itertext()).strip()
                              for e in el.findall("sevenChrDef/extension")}))
    for child in el.findall("diag"):
        _add_diag(nodes, child, code, depth + 1)


@lru_cache(maxsize=1)
def load_tree(path: Path | None = None) -> ICD10Tree:
    path = path or load_config()["paths"]["icd10_dir"] / "icd10cm_tabular_2025.xml"
    root, nodes = ET.parse(path).getroot(), {}
    for ch in root.findall("chapter"):
        cid = f"Ch{ch.findtext('name')}"
        _add(nodes, Node(cid, ch.findtext("desc"), "chapter", None, notes=_notes(ch)))
        for sec in ch.findall("section"):
            sid = sec.get("id") if "-" in sec.get("id") else f"{sec.get('id')}-{sec.get('id')}"  # 38 single-category blocks
            _add(nodes, Node(sid, sec.findtext("desc"), "block", cid, notes=_notes(sec)))
            for diag in sec.findall("diag"):
                _add_diag(nodes, diag, sid, 0)
    return ICD10Tree(nodes)
