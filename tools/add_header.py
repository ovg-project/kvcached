#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import List

# Extensions we handle
PY_EXTS = {".py"}
C_EXTS  = {".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh", ".ipp"}

# Tokens that indicate an existing (possibly incomplete) header we should replace
HEADER_TOKENS = ("spdx", "license", "copyright")

ROOT = Path(__file__).resolve().parent.parent
HEADER_FILE = ROOT / ".license-header.txt"
if not HEADER_FILE.exists():
    sys.stderr.write("ERROR: .license-header.txt not found in repo root\n")
    sys.exit(1)

HEADER_LINES_RAW = HEADER_FILE.read_text(encoding="utf-8").strip().splitlines()

def detect_newline(text: str) -> str:
    return "\r\n" if "\r\n" in text and "\n" in text.replace("\r\n", "") else ("\r\n" if "\r\n" in text else "\n")

def wrap_header(ext: str, newline: str) -> str:
    """Return the header string wrapped for the given extension, with a blank line after."""
    if ext in PY_EXTS:
        lines = [f"# {ln}" if ln else "#" for ln in HEADER_LINES_RAW]
    elif ext in C_EXTS:
        # /** ... */ style for C/C++
        body = [f" * {ln}" if ln else " *" for ln in HEADER_LINES_RAW]
        lines = ["/**", *body, " */"]
    else:
        lines = HEADER_LINES_RAW[:]  # fallback (shouldn't happen due to files regex)
    return newline.join(lines) + (newline * 2)

def _is_token_line(line: str) -> bool:
    normalized_line = line.strip().lower()
    return any(token in normalized_line for token in HEADER_TOKENS)

def _find_py_insert_index(lines: List[str]) -> int:
    """For Python: place header after shebang and encoding lines."""
    i = 0
    if i < len(lines) and lines[i].startswith("#!"):
        i += 1
    if i < len(lines) and ("coding:" in lines[i] or "coding=" in lines[i]):
        i += 1
    return i

def _strip_block(lines: List[str], start: int, end: int) -> List[str]:
    # Remove lines[start:end] and also a single trailing blank line if present
    out = lines[:start] + lines[end:]
    if start < len(out) and out[start].strip() == "":
        out.pop(start)
    return out

def strip_existing_header_py(lines: List[str], start_idx: int) -> List[str]:
    """
    Remove a top-of-file Python header block if it looks like license/SPDX/copyright.
    - Consecutive lines starting with '#' (with optional single blank before code)
    - Only removed if at least one of those lines contains HEADER_TOKENS
    """
    i = start_idx
    j = i
    # Allow at most one initial blank line (rare, but be tolerant)
    if j < len(lines) and lines[j].strip() == "":
        j += 1
    k = j
    found_hash = False
    while k < len(lines):
        s = lines[k].lstrip()
        if s.startswith("#"):
            found_hash = True
            k += 1
            continue
        break

    if not found_hash:
        return lines

    block_lines = lines[j:k]
    if any(_is_token_line(ln) for ln in block_lines):
        return _strip_block(lines, j, k)
    return lines

def strip_existing_header_c(lines: List[str]) -> List[str]:
    """
    Remove a C/C++ style header at the very top if it looks like license/SPDX/copyright.
    Supports:
      - /** ... */
      - /* ...  */
      - consecutive // ... lines
    Only removed if the block contains HEADER_TOKENS.
    """
    i = 0
    # Tolerate one leading blank line
    if i < len(lines) and lines[i].strip() == "":
        i += 1

    # 1) /* ... */ or /** ... */
    if i < len(lines) and lines[i].lstrip().startswith("/*"):
        j = i + 1
        has_tokens = _is_token_line(lines[i])
        while j < len(lines):
            has_tokens = has_tokens or _is_token_line(lines[j])
            if "*/" in lines[j]:
                # candidate block i..j inclusive
                if has_tokens:
                    return _strip_block(lines, i, j + 1)
                break
            j += 1
        # If no closing */, fall through to // block check (don't delete partial comments)

    # 2) //-style consecutive lines
    j = i
    got = False
    has_tokens = False
    while j < len(lines):
        s = lines[j].lstrip()
        if s.startswith("//"):
            got = True
            if _is_token_line(lines[j]):
                has_tokens = True
            j += 1
            continue
        break

    if got and has_tokens:
        return _strip_block(lines, i, j)

    return lines

def process_file(p: Path) -> bool:
    try:
        text = p.read_text(encoding="utf-8")
    except Exception:
        return False

    newline = detect_newline(text)
    lines = text.splitlines()

    ext = p.suffix.lower()
    if ext in PY_EXTS:
        idx = _find_py_insert_index(lines)
        # First strip an existing header (if any) after shebang/encoding
        lines = strip_existing_header_py(lines, idx)
        # Recompute idx in case lines shifted
        idx = _find_py_insert_index(lines)
        header = wrap_header(ext, newline)
        new_text = newline.join(lines[:idx] + [header.rstrip("\n")] + lines[idx:])
    elif ext in C_EXTS:
        # Strip any existing top-of-file header
        lines = strip_existing_header_c(lines)
        header = wrap_header(ext, newline)
        new_text = header + newline.join(lines) if lines else header
    else:
        return False  # should not happen due to file filter

    # Preserve final newline if original had it; our header already includes one blank line
    if text.endswith(("\r\n", "\n")) and not new_text.endswith(newline):
        new_text += newline

    if new_text == text:
        return False
    p.write_text(new_text, encoding="utf-8", newline="")
    return True

def main():
    for fname in sys.argv[1:]:
        p = Path(fname)
        if p.is_file() and p.suffix.lower() in (PY_EXTS | C_EXTS):
            process_file(p)
    # Exit 0 is fine; pre-commit will re-stage as needed
    sys.exit(0)

if __name__ == "__main__":
    main()
