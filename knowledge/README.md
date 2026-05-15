# pfmr Knowledge Graph

This directory contains the pfmr knowledge graph — a curated, append-only
database of facts about Python packages and their Flatpak build dependencies.

## Structure

```
knowledge/
  nodes/
    packages.toml     — Python package nodes (pypi name, build backend, ...)
    libraries.toml    — Native library nodes (soname, pkgconfig name, ...)
    extensions.toml   — SDK Extension nodes
    sdks.toml         — Flatpak SDK / runtime nodes
  edges/
    requires_pkgconfig.toml   — package → pkgconfig dependency
    requires_library.toml     — package → shared library dependency
    requires_extension.toml   — package → SDK extension requirement
    provides_pkgconfig.toml   — sdk/extension → pkgconfig it provides
    provides_library.toml     — sdk/extension → library it provides
    triggers_extension.toml   — package → extension trigger
    belongs_to_sdk.toml       — library → sdk it belongs to
```

## Contributing

### Adding a new package dependency

Edit `edges/requires_pkgconfig.toml`:

```toml
[[edge]]
from = "cryptography"           # canonical PyPI name (lowercase, hyphens)
to = "openssl"                  # pkgconfig name
relation = "requires_pkgconfig"
confidence = 1.0                # 1.0 = confirmed by sandbox probe
source = "sandbox:org.freedesktop.Sdk/24.08"
updated = "2025-05-14"
```

### Adding a new library node

Edit `nodes/libraries.toml`:

```toml
[libusb]
node_type = "library"
pkgconfig = "libusb-1.0"
soname = "libusb-1.0.so.0"
```

### Confidence levels

| Value | Meaning |
|-------|---------|
| 1.0   | Confirmed by live sandbox probe |
| 0.9   | Confirmed by successful Flathub build |
| 0.8   | Inferred from Flathub manifest analysis |
| 0.7   | Inferred from wheel ELF analysis |
| 0.5   | Heuristic (wheel tag, package name pattern) |

### Automated update workflow

```bash
# Mine Flathub for new manifests
pfmr learn flathub --limit 500 --export

# Analyze a specific app
pfmr learn manifest /path/to/org.gnome.MyApp.yaml --export

# Ingest a probe report
pfmr learn ingest probe-report.json --package cryptography --export

# View statistics
pfmr learn stats

# Export to recipe files
pfmr learn export
```

The `pfmr learn flathub` command is designed to be run in a GitHub Actions
workflow on a schedule, automatically submitting PRs with new knowledge.

## Design principles

- **Append-only**: Entries with `confirmed = true` are never removed by automated tools
- **Git-native**: Plain TOML, one entry per line, easy to review in PRs
- **Transparent**: Every edge has a `source` and `updated` field
- **Composable**: The graph can be queried without running pfmr's resolver