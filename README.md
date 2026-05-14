# pfmr — Python Flatpak Manifest Resolver

Modern replacement for `flatpak-pip-generator`.

## Phase 1 (implemented)

| Feature | Status |
|---|---|
| UV Resolver Engine | ✅ |
| Flatpak manifest generation (JSON + YAML) | ✅ |
| Local recipe database | ✅ |
| Build-backend detection | ✅ |
| Native package detection (heuristic) | ✅ |
| SDK Extension auto-declaration (Rust) | ✅ |
| Pure-python batching | ✅ |
| Deterministic lockfile hash | ✅ |

## Structure

pfmr/
├── pfmr/
│   ├── models.py            # Tipos compartilhados (ResolvedPackage, FlatpakModule, etc.)
│   ├── pipeline.py          # Orquestrador — conecta todos os componentes
│   ├── cli.py               # CLI typer (pfmr resolve / generate / recipes)
│   ├── resolvers/
│   │   └── uv_resolver.py   # UV Resolver Engine
│   ├── generators/
│   │   └── manifest.py      # Flatpak Manifest Generator (JSON + YAML)
│   └── recipes/
│       └── db.py            # RecipeDB — banco local de recipes
├── recipes/
│   ├── native/              # libusb, libffi, hidapi, libvips, openblas
│   └── extensions/          # rust-stable
└── tests/
    └── test_phase1.py       # 22 testes

## Roadmap

- **Phase 2** — ELF analysis, SDK capability resolver, SDK extension resolver
- **Phase 3** — Build sandbox probing, auto-learning, knowledge graph
- **Phase 4** — Distributed registry, Flathub integration, CI intelligence

## Installation

```bash
pip install -e ".[dev]"
```

Requires `uv` on `$PATH`

## Usage

```bash
# Resolve a package and show summary
pfmr resolve requests

# Resolve from pyproject.toml
pfmr resolve pyproject.toml

# Generate a Flatpak manifest
pfmr generate pyproject.toml -o modules.yaml

# Generate JSON
pfmr generate "numpy==1.26.4" -f json -o numpy-module.json

# List available recipes
pfmr recipes list

# Show a recipe
pfmr recipes show libusb
```

## Recipe format

```yaml
id: libusb
provides:
  - libusb-1.0.so
  - libusb-1.0.so.0
pkgconfig:
  - libusb-1.0
headers:
  - libusb.h
aliases:
  - usb
buildsystem: autotools        # autotools | cmake | meson | simple
source:
  type: archive
  url: https://github.com/libusb/libusb/releases/download/v1.0.27/libusb-1.0.27.tar.bz2
  sha256: ffaa41d7...
config-opts:
  - --disable-static
cleanup:
  - /include
  - /lib/pkgconfig
```

## Architecture (Phase 1)

```
Input (pyproject.toml | requirements.txt | package spec)
    ↓
UVResolver          →  ResolutionResult
    ↓
RecipeDB lookup     →  attach NativeRecipes
    ↓
ManifestGenerator   →  FlatpakManifest
    ↓
YAML / JSON output
```
