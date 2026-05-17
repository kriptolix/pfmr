"""
pfmr.learn — standalone learning and import system.

Completely decoupled from the resolver pipeline.
Mines Flathub, imports shared-modules, probes SDKs, and writes
results directly into recipes/ and data/ — no knowledge graph.
"""
from pfmr.learn.manifest import ManifestAnalyzer, ManifestAnalysis
from pfmr.learn.flathub import FlathubMiner, MineResult, MineProgress
from pfmr.learn.shared_modules import SharedModulesImporter, ImportReport
from pfmr.learn.sdk_probe import SDKProber, ProbeResult
from pfmr.learn.sandbox import SandboxLearner
from pfmr.learn.exporter import Exporter, ExportReport

__all__ = [
    "ManifestAnalyzer", "ManifestAnalysis",
    "FlathubMiner", "MineResult", "MineProgress",
    "SharedModulesImporter", "ImportReport",
    "SDKProber", "ProbeResult",
    "SandboxLearner",
    "Exporter", "ExportReport",
]