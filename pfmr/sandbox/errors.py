"""
pfmr.sandbox.errors
~~~~~~~~~~~~~~~~~~~~
Parses raw stderr / stdout from inside the Flatpak build sandbox and
produces normalised SandboxError objects.

Recognised patterns (in priority order):
  - Missing shared library  (ldd / linker output)
  - Missing header file      (compiler fatal error)
  - Missing pkg-config entry (configure / meson / cmake output)
  - Missing executable       (command not found)
  - Python ImportError       (python -c / pip install output)
  - pip / uv install failure (package not found, version conflict)
  - Generic build failure    (non-zero exit with unrecognised message)
"""
from __future__ import annotations

import re
from typing import Optional

from pfmr.models import SandboxError, SandboxErrorType

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# ldd output:  "    libusb-1.0.so.0 => not found"
_P_LDD_NOT_FOUND = re.compile(
    r"^\s*(?P<lib>\S+\.so\S*)\s*=>\s*not found", re.MULTILINE
)

# linker:  "cannot find -lusb-1.0"  or  "error while loading shared libraries: libfoo.so"
_P_LINKER_NOT_FOUND = re.compile(
    r"(?:cannot find -l(?P<lib1>\S+)|"
    r"error while loading shared libraries:\s*(?P<lib2>[^\s:]+\.so[^\s:]*))",
    re.IGNORECASE,
)

# GCC/Clang fatal:  "fatal error: openssl/ssl.h: No such file or directory"
_P_MISSING_HEADER = re.compile(
    r"fatal error:\s*(?P<header>[^\s:]+\.h[^:]*?):\s*No such file",
    re.IGNORECASE,
)

# pkg-config:  "Package openssl was not found in the pkg-config search path"
#              "No package 'libffi' found"
#              "Could not find dependency 'foo' ..."
_P_PKGCONFIG_NOT_FOUND = re.compile(
    r"(?:Package\s+(?P<pkg1>\S+)\s+was not found|"
    r"No package\s+'?(?P<pkg2>[^'>\s]+)'?\s+found|"
    r"Could not find dependency\s+'?(?P<pkg3>[^'>\s]+)'?)",
    re.IGNORECASE,
)

# "command not found" / "No such file or directory" for executables
_P_EXEC_NOT_FOUND = re.compile(
    r"(?P<cmd>\S+):\s*(?:command not found|No such file or directory)",
    re.IGNORECASE,
)

# Python ImportError
_P_IMPORT_ERROR = re.compile(
    r"(?:ModuleNotFoundError|ImportError):\s*(?:No module named\s*'?(?P<mod>[^'\";\n]+)'?)",
    re.IGNORECASE,
)

# pip / uv install failure
_P_PIP_NOT_FOUND = re.compile(
    r"(?:ERROR: No matching distribution found for\s+(?P<dist>[^\s=<>!]+)|"
    r"error: Package\s+'?(?P<pkg>[^'\s]+)'?\s+not found|"
    r"Could not find a version that satisfies the requirement\s+(?P<req>[^\s=<>!]+))",
    re.IGNORECASE,
)

# Meson "Dependency X not found"
_P_MESON_DEP = re.compile(
    r"Dependency\s+(?P<dep>\S+)\s+found:\s+NO",
    re.IGNORECASE,
)

# CMake "Could NOT find <Foo>"
_P_CMAKE_NOT_FOUND = re.compile(
    r"Could NOT find\s+(?P<dep>\S+)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_errors(
    stderr: str,
    stdout: str = "",
    ldd_output: str = "",
    context: str = "",
) -> list[SandboxError]:
    """
    Parse all available output and return a deduplicated list of SandboxError.
    """
    errors: list[SandboxError] = []
    seen: set[tuple[str, str]] = set()   # (error_type, missing)

    def _add(error: SandboxError) -> None:
        key = (error.error_type.value, error.missing.lower())
        if key not in seen:
            errors.append(error)
            seen.add(key)

    # --- ldd output ---
    for m in _P_LDD_NOT_FOUND.finditer(ldd_output):
        _add(SandboxError(
            error_type=SandboxErrorType.MISSING_NATIVE_DEP,
            missing=m.group("lib"),
            source="ldd",
            context=context,
            raw_line=m.group(0).strip(),
        ))

    # --- stderr + stdout (combined for patterns that appear in either) ---
    combined = stderr + "\n" + stdout

    for m in _P_LINKER_NOT_FOUND.finditer(combined):
        lib = m.group("lib1") or m.group("lib2") or ""
        if lib:
            _add(SandboxError(
                error_type=SandboxErrorType.MISSING_NATIVE_DEP,
                missing=lib,
                source="stderr",
                context=context,
                raw_line=m.group(0).strip(),
            ))

    for m in _P_MISSING_HEADER.finditer(combined):
        _add(SandboxError(
            error_type=SandboxErrorType.MISSING_HEADER,
            missing=m.group("header").strip(),
            source="stderr",
            context=context,
            raw_line=m.group(0).strip(),
        ))

    for m in _P_PKGCONFIG_NOT_FOUND.finditer(combined):
        pkg = m.group("pkg1") or m.group("pkg2") or m.group("pkg3") or ""
        if pkg:
            _add(SandboxError(
                error_type=SandboxErrorType.MISSING_PKGCONFIG,
                missing=pkg.strip(),
                source="stderr",
                context=context,
                raw_line=m.group(0).strip(),
            ))

    for m in _P_MESON_DEP.finditer(combined):
        _add(SandboxError(
            error_type=SandboxErrorType.MISSING_PKGCONFIG,
            missing=m.group("dep").strip(),
            source="stderr",
            context=context,
            raw_line=m.group(0).strip(),
        ))

    for m in _P_CMAKE_NOT_FOUND.finditer(combined):
        dep = m.group("dep").strip()
        # Skip generic cmake internal deps (uppercase, short)
        if len(dep) > 2:
            _add(SandboxError(
                error_type=SandboxErrorType.MISSING_PKGCONFIG,
                missing=dep,
                source="stderr",
                context=context,
                raw_line=m.group(0).strip(),
            ))

    for m in _P_IMPORT_ERROR.finditer(combined):
        mod = m.group("mod").strip().split(".")[0]  # top-level module only
        if mod:
            _add(SandboxError(
                error_type=SandboxErrorType.IMPORT_ERROR,
                missing=mod,
                source="import",
                context=context,
                raw_line=m.group(0).strip(),
            ))

    for m in _P_PIP_NOT_FOUND.finditer(combined):
        pkg = m.group("dist") or m.group("pkg") or m.group("req") or ""
        if pkg:
            _add(SandboxError(
                error_type=SandboxErrorType.MISSING_PYTHON_PKG,
                missing=pkg.strip(),
                source="stderr",
                context=context,
                raw_line=m.group(0).strip(),
            ))

    for m in _P_EXEC_NOT_FOUND.finditer(combined):
        cmd = m.group("cmd").strip()
        # Filter noise — single-char tokens and common shell words
        if len(cmd) > 2 and "/" not in cmd:
            _add(SandboxError(
                error_type=SandboxErrorType.MISSING_EXECUTABLE,
                missing=cmd,
                source="stderr",
                context=context,
                raw_line=m.group(0).strip(),
            ))

    return errors