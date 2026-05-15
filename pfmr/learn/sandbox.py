"""
pfmr.learn.sandbox
~~~~~~~~~~~~~~~~~~~
SandboxLearner — ingests a SandboxProbeReport (from a successful or failed
build probe) and updates the KnowledgeGraph with high-confidence facts.

A successful build is the most reliable source of truth:
  - Package installed → confirmed requires_native + native_deps
  - ldd clean → confirmed no external deps beyond SDK
  - Probe errors → confirmed missing deps

A failed build is also valuable:
  - Missing header / lib / pkgconfig → confirmed dep not in SDK
  - These facts prevent re-probing the same thing repeatedly

Confidence values:
  1.0 — successful sandbox install + import
  0.9 — successful install, import failed (may be env issue)
  0.8 — failed install with clear error (known-missing dep)
  0.5 — failed install with generic error

Completely standalone — does not import from pfmr.pipeline.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

from packaging.utils import canonicalize_name

from pfmr.learn.graph import KGEdge, KGNode, KnowledgeGraph, Rel
from pfmr.models import SandboxErrorType, SandboxProbeReport
from pfmr.utils.logging import get_logger

logger = get_logger(__name__)


class SandboxLearner:
    """
    Updates a KnowledgeGraph from SandboxProbeReport data.

    Usage (standalone)::

        kg = KnowledgeGraph(Path("knowledge/"))
        learner = SandboxLearner(kg)
        learner.ingest(report, package_name="cryptography")
        kg.save()
    """

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(
        self,
        report: SandboxProbeReport,
        package_name: Optional[str] = None,
        sdk_id: str = "org.freedesktop.Sdk",
        sdk_version: str = "24.08",
    ) -> int:
        """
        Ingest a probe report into the knowledge graph.

        Args:
            report:       The SandboxProbeReport to learn from.
            package_name: If given, binds errors to this specific package.
            sdk_id:       The SDK that was used during the probe.
            sdk_version:  The SDK version.

        Returns:
            Number of new facts added to the graph.
        """
        if not report.ran:
            logger.debug("Skipping ingestion — probe did not run")
            return 0

        added = 0
        source = f"sandbox:{sdk_id}/{sdk_version}"
        today = str(date.today())

        # --- Ensure SDK node exists ---
        sdk_node_id = f"{sdk_id}//{sdk_version}"
        self.graph.add_node(KGNode(
            id=sdk_node_id,
            node_type="sdk",
            attrs={"sdk_id": sdk_id, "sdk_version": sdk_version},
        ))

        # --- Process each error ---
        for err in report.errors:
            context_pkg = package_name or err.context.split(" ")[0]
            canonical = canonicalize_name(context_pkg) if context_pkg else None

            # Ensure the package node exists
            if canonical:
                self.graph.add_node(KGNode(
                    id=canonical,
                    node_type="package",
                    attrs={"pypi_name": context_pkg},
                ))

            if err.error_type == SandboxErrorType.MISSING_NATIVE_DEP:
                lib_id = _normalise_soname(err.missing)
                self.graph.add_node(KGNode(
                    id=lib_id, node_type="library",
                    attrs={"soname": err.missing},
                ))
                if canonical:
                    added += int(self.graph.add_edge(KGEdge(
                        from_id=canonical, to_id=lib_id,
                        relation=Rel.REQUIRES_LIBRARY,
                        confidence=0.8,
                        source=source, updated=today,
                    )))

            elif err.error_type == SandboxErrorType.MISSING_HEADER:
                hdr_id = err.missing.replace("/", "_").replace(".", "_")
                self.graph.add_node(KGNode(
                    id=hdr_id, node_type="library",
                    attrs={"header": err.missing},
                ))
                if canonical:
                    added += int(self.graph.add_edge(KGEdge(
                        from_id=canonical, to_id=hdr_id,
                        relation=Rel.REQUIRES_LIBRARY,
                        confidence=0.8,
                        source=source, updated=today,
                    )))

            elif err.error_type == SandboxErrorType.MISSING_PKGCONFIG:
                pc_id = err.missing
                self.graph.add_node(KGNode(
                    id=pc_id, node_type="library",
                    attrs={"pkgconfig": err.missing},
                ))
                if canonical:
                    added += int(self.graph.add_edge(KGEdge(
                        from_id=canonical, to_id=pc_id,
                        relation=Rel.REQUIRES_PKGCONFIG,
                        confidence=0.8,
                        source=source, updated=today,
                    )))
                # Also record that the SDK does NOT provide this
                added += int(self.graph.add_edge(KGEdge(
                    from_id=sdk_node_id, to_id=pc_id,
                    relation="missing_in_sdk",
                    confidence=0.9,
                    source=source, updated=today,
                )))

        # --- Learn from successful installs (packages that had no errors) ---
        if report.ran and not report.errors and package_name:
            canonical = canonicalize_name(package_name)
            self.graph.add_node(KGNode(
                id=canonical,
                node_type="package",
                attrs={
                    "pypi_name": package_name,
                    "confirmed": True,
                    "sdk_sufficient": True,
                },
            ))
            added += 1

        logger.info(
            "Ingested probe report: %d new facts added (package=%s, errors=%d)",
            added, package_name, len(report.errors),
        )
        return added

    def ingest_successful_build(
        self,
        package_name: str,
        native_deps: list[str],
        required_extensions: list[str],
        sdk_id: str = "org.freedesktop.Sdk",
        sdk_version: str = "24.08",
    ) -> int:
        """
        Record a confirmed successful build with known deps.
        This is the highest-confidence learning path.
        """
        canonical = canonicalize_name(package_name)
        today = str(date.today())
        source = f"sandbox:{sdk_id}/{sdk_version}"
        added = 0

        self.graph.add_node(KGNode(
            id=canonical,
            node_type="package",
            attrs={
                "pypi_name": package_name,
                "confirmed": True,
                "sdk_sufficient": not bool(native_deps),
            },
        ))

        for dep in native_deps:
            dep_id = dep
            self.graph.add_node(KGNode(
                id=dep_id, node_type="library", attrs={"pkgconfig": dep}
            ))
            added += int(self.graph.add_edge(KGEdge(
                from_id=canonical, to_id=dep_id,
                relation=Rel.REQUIRES_PKGCONFIG,
                confidence=1.0,
                source=source, updated=today,
            )))

        for ext in required_extensions:
            self.graph.add_node(KGNode(
                id=ext, node_type="extension", attrs={"extension_id": ext}
            ))
            added += int(self.graph.add_edge(KGEdge(
                from_id=canonical, to_id=ext,
                relation=Rel.REQUIRES_EXTENSION,
                confidence=1.0,
                source=source, updated=today,
            )))

        return added


def _normalise_soname(soname: str) -> str:
    """Strip version suffix: libssl.so.3 → libssl"""
    import re
    base = re.sub(r"\.so(\..+)?$", "", soname)
    return base.lstrip("lib") if base.startswith("lib") else base