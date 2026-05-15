"""
pfmr.learn.graph
~~~~~~~~~~~~~~~~~
KnowledgeGraph — the central knowledge store for the pfmr learning system.

Design principles:
  - Plain text TOML files, one per entity type → git-friendly, human-editable
  - Append-only: new facts are added; existing confirmed facts are never deleted
  - Confidence score on every edge (0.0–1.0):
      1.0 = confirmed by live sandbox build
      0.8 = inferred from Flathub manifest analysis
      0.5 = heuristic / wheel tag
  - Fully decoupled from the resolver pipeline — no pfmr.pipeline imports

Storage layout (all under <knowledge_dir>/):
  nodes/
    packages.toml     — Python package nodes
    libraries.toml    — native library nodes
    extensions.toml   — SDK extension nodes
    sdks.toml         — SDK / runtime nodes
  edges/
    requires.toml     — package → library / extension / pkgconfig
    provides.toml     — sdk/extension → library / pkgconfig
    triggers.toml     — package → extension (build-time trigger)
    belongs_to.toml   — library → sdk (already in SDK)

Node format (packages.toml):
  [cryptography]
  canonical_name = "cryptography"
  pypi_name = "cryptography"
  build_backend = "maturin"
  confirmed = true
  source = "flathub:org.gnome.Crypto"
  updated = "2025-05-14"

Edge format (requires.toml):
  [[edge]]
  from = "cryptography"
  to = "openssl"
  relation = "requires_pkgconfig"
  confidence = 1.0
  source = "sandbox:probe"
  updated = "2025-05-14"
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import yaml

from pfmr.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class KGNode:
    """A node in the knowledge graph."""
    id: str
    node_type: str          # "package" | "library" | "extension" | "sdk"
    attrs: dict = field(default_factory=dict)

    def to_toml_entry(self) -> str:
        lines = [f"[{self.id}]"]
        lines.append(f'node_type = "{self.node_type}"')
        for k, v in sorted(self.attrs.items()):
            if isinstance(v, bool):
                lines.append(f"{k} = {'true' if v else 'false'}")
            elif isinstance(v, (int, float)):
                lines.append(f"{k} = {v}")
            elif isinstance(v, list):
                items = ", ".join(f'"{i}"' for i in v)
                lines.append(f"{k} = [{items}]")
            else:
                escaped = str(v).replace('"', '\\"')
                lines.append(f'{k} = "{escaped}"')
        return "\n".join(lines)


@dataclass
class KGEdge:
    """A directed, typed edge between two nodes."""
    from_id: str
    to_id: str
    relation: str           # "requires_pkgconfig" | "requires_library" | "requires_extension"
                            # "provides_pkgconfig" | "provides_library" | "triggers_extension"
                            # "belongs_to_sdk"
    confidence: float = 0.8
    source: str = ""        # e.g. "flathub:org.gnome.App" | "sandbox:probe" | "heuristic"
    updated: str = ""       # ISO date

    def __post_init__(self):
        if not self.updated:
            self.updated = str(date.today())

    def to_toml_entry(self) -> str:
        return (
            "[[edge]]\n"
            f'from = "{self.from_id}"\n'
            f'to = "{self.to_id}"\n'
            f'relation = "{self.relation}"\n'
            f"confidence = {self.confidence}\n"
            f'source = "{self.source}"\n'
            f'updated = "{self.updated}"'
        )


# ---------------------------------------------------------------------------
# Edge relation constants
# ---------------------------------------------------------------------------

class Rel:
    REQUIRES_PKGCONFIG  = "requires_pkgconfig"
    REQUIRES_LIBRARY    = "requires_library"
    REQUIRES_EXTENSION  = "requires_extension"
    PROVIDES_PKGCONFIG  = "provides_pkgconfig"
    PROVIDES_LIBRARY    = "provides_library"
    TRIGGERS_EXTENSION  = "triggers_extension"
    BELONGS_TO_SDK      = "belongs_to_sdk"


# ---------------------------------------------------------------------------
# KnowledgeGraph
# ---------------------------------------------------------------------------

class KnowledgeGraph:
    """
    Append-only knowledge graph stored as plain TOML files.

    Usage (standalone — no pipeline dependency)::

        kg = KnowledgeGraph(Path("knowledge/"))
        kg.add_node(KGNode("cryptography", "package", {"build_backend": "maturin"}))
        kg.add_edge(KGEdge("cryptography", "openssl", Rel.REQUIRES_PKGCONFIG,
                            confidence=1.0, source="sandbox:probe"))
        kg.save()

        # Query
        deps = kg.edges_from("cryptography", relation=Rel.REQUIRES_PKGCONFIG)
    """

    def __init__(self, knowledge_dir: Path):
        self.knowledge_dir = knowledge_dir
        self._nodes: dict[str, KGNode] = {}          # id → node
        self._edges: list[KGEdge] = []
        self._edge_keys: set[tuple] = set()           # dedup key
        self._load()

    # ------------------------------------------------------------------
    # Public API — write
    # ------------------------------------------------------------------

    def add_node(self, node: KGNode, overwrite: bool = False) -> bool:
        """
        Add a node. If the node already exists and overwrite=False, merges
        new attrs without removing existing ones. Returns True if changed.
        """
        existing = self._nodes.get(node.id)
        if existing is None:
            self._nodes[node.id] = node
            return True
        if overwrite:
            self._nodes[node.id] = node
            return True
        # Merge: add keys absent from existing
        changed = False
        for k, v in node.attrs.items():
            if k not in existing.attrs:
                existing.attrs[k] = v
                changed = True
        return changed

    def add_edge(self, edge: KGEdge) -> bool:
        """
        Add a directed edge. Deduplicates by (from, to, relation).
        If the same edge exists with lower confidence, updates confidence.
        Returns True if the graph changed.
        """
        key = (edge.from_id, edge.to_id, edge.relation)
        # Check for existing edge with same key
        for existing in self._edges:
            if (existing.from_id, existing.to_id, existing.relation) == key:
                if edge.confidence > existing.confidence:
                    existing.confidence = edge.confidence
                    existing.source = edge.source
                    existing.updated = edge.updated
                    return True
                return False
        self._edges.append(edge)
        self._edge_keys.add(key)
        return True

    def save(self) -> None:
        """Persist all nodes and edges to TOML files."""
        self._ensure_dirs()
        self._save_nodes()
        self._save_edges()
        logger.info(
            "Knowledge graph saved: %d nodes, %d edges → %s",
            len(self._nodes), len(self._edges), self.knowledge_dir,
        )

    # ------------------------------------------------------------------
    # Public API — read
    # ------------------------------------------------------------------

    def node(self, node_id: str) -> Optional[KGNode]:
        return self._nodes.get(node_id)

    def nodes_of_type(self, node_type: str) -> list[KGNode]:
        return [n for n in self._nodes.values() if n.node_type == node_type]

    def edges_from(
        self,
        node_id: str,
        relation: Optional[str] = None,
    ) -> list[KGEdge]:
        return [
            e for e in self._edges
            if e.from_id == node_id and (relation is None or e.relation == relation)
        ]

    def edges_to(
        self,
        node_id: str,
        relation: Optional[str] = None,
    ) -> list[KGEdge]:
        return [
            e for e in self._edges
            if e.to_id == node_id and (relation is None or e.relation == relation)
        ]

    def requires(self, package_id: str) -> list[str]:
        """Return all pkgconfig/library/extension ids required by a package."""
        return [
            e.to_id for e in self._edges
            if e.from_id == package_id and e.relation in (
                Rel.REQUIRES_PKGCONFIG, Rel.REQUIRES_LIBRARY, Rel.REQUIRES_EXTENSION,
            )
        ]

    def what_provides(self, dep_id: str) -> list[str]:
        """Return SDK/extension ids that provide a given dep."""
        return [
            e.from_id for e in self._edges
            if e.to_id == dep_id and e.relation in (
                Rel.PROVIDES_PKGCONFIG, Rel.PROVIDES_LIBRARY, Rel.BELONGS_TO_SDK,
            )
        ]

    def stats(self) -> dict:
        by_type: dict[str, int] = {}
        for n in self._nodes.values():
            by_type[n.node_type] = by_type.get(n.node_type, 0) + 1
        by_rel: dict[str, int] = {}
        for e in self._edges:
            by_rel[e.relation] = by_rel.get(e.relation, 0) + 1
        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "nodes_by_type": by_type,
            "edges_by_relation": by_rel,
        }

    def __len__(self) -> int:
        return len(self._nodes) + len(self._edges)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _ensure_dirs(self) -> None:
        (self.knowledge_dir / "nodes").mkdir(parents=True, exist_ok=True)
        (self.knowledge_dir / "edges").mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        if not self.knowledge_dir.exists():
            return
        self._load_nodes()
        self._load_edges()
        logger.debug(
            "Loaded knowledge graph: %d nodes, %d edges",
            len(self._nodes), len(self._edges),
        )

    def _load_nodes(self) -> None:
        nodes_dir = self.knowledge_dir / "nodes"
        if not nodes_dir.exists():
            return
        for toml_file in sorted(nodes_dir.glob("*.toml")):
            try:
                with open(toml_file, "rb") as f:
                    data = tomllib.load(f)
                for node_id, attrs in data.items():
                    if not isinstance(attrs, dict):
                        continue
                    node_type = attrs.pop("node_type", "unknown")
                    self._nodes[node_id] = KGNode(
                        id=node_id,
                        node_type=node_type,
                        attrs=dict(attrs),
                    )
            except Exception as exc:
                logger.warning("Failed to load node file %s: %s", toml_file, exc)

    def _load_edges(self) -> None:
        edges_dir = self.knowledge_dir / "edges"
        if not edges_dir.exists():
            return
        for toml_file in sorted(edges_dir.glob("*.toml")):
            try:
                with open(toml_file, "rb") as f:
                    data = tomllib.load(f)
                for raw in data.get("edge", []):
                    edge = KGEdge(
                        from_id=raw["from"],
                        to_id=raw["to"],
                        relation=raw["relation"],
                        confidence=raw.get("confidence", 0.8),
                        source=raw.get("source", ""),
                        updated=raw.get("updated", ""),
                    )
                    key = (edge.from_id, edge.to_id, edge.relation)
                    if key not in self._edge_keys:
                        self._edges.append(edge)
                        self._edge_keys.add(key)
            except Exception as exc:
                logger.warning("Failed to load edge file %s: %s", toml_file, exc)

    def _save_nodes(self) -> None:
        # Group nodes by type into separate files
        by_type: dict[str, list[KGNode]] = {}
        for node in self._nodes.values():
            by_type.setdefault(node.node_type, []).append(node)

        type_to_file = {
            "package":   "packages.toml",
            "library":   "libraries.toml",
            "extension": "extensions.toml",
            "sdk":       "sdks.toml",
        }
        nodes_dir = self.knowledge_dir / "nodes"
        for node_type, nodes in by_type.items():
            filename = type_to_file.get(node_type, f"{node_type}.toml")
            path = nodes_dir / filename
            lines = [
                "# pfmr knowledge graph — node file",
                f"# type: {node_type}",
                f"# entries: {len(nodes)}",
                "# DO NOT EDIT manually entries marked confirmed = true",
                "",
            ]
            for node in sorted(nodes, key=lambda n: n.id):
                lines.append(node.to_toml_entry())
                lines.append("")
            path.write_text("\n".join(lines))

    def _save_edges(self) -> None:
        # Group edges by relation type
        by_rel: dict[str, list[KGEdge]] = {}
        for edge in self._edges:
            by_rel.setdefault(edge.relation, []).append(edge)

        edges_dir = self.knowledge_dir / "edges"
        for relation, edges in by_rel.items():
            path = edges_dir / f"{relation}.toml"
            lines = [
                "# pfmr knowledge graph — edge file",
                f"# relation: {relation}",
                f"# entries: {len(edges)}",
                "",
            ]
            for edge in sorted(edges, key=lambda e: (e.from_id, e.to_id)):
                lines.append(edge.to_toml_entry())
                lines.append("")
            path.write_text("\n".join(lines))