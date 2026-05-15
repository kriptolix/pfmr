"""
pfmr.learn — standalone learning and knowledge graph system.

Completely decoupled from the resolver pipeline.
Can be used as an independent tool to mine Flathub, ingest probe reports,
and export updated recipes and hints back to the repository.
"""
from pfmr.learn.graph import KnowledgeGraph, KGNode, KGEdge, Rel
from pfmr.learn.manifest import ManifestAnalyzer, ManifestAnalysis
from pfmr.learn.flathub import FlathubMiner, MineResult
from pfmr.learn.sandbox import SandboxLearner
from pfmr.learn.exporter import Exporter, ExportReport

__all__ = [
    "KnowledgeGraph", "KGNode", "KGEdge", "Rel",
    "ManifestAnalyzer", "ManifestAnalysis",
    "FlathubMiner", "MineResult",
    "SandboxLearner",
    "Exporter", "ExportReport",
]