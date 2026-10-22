"""Fail if any tracked text file contains an em dash, en dash or similar.

    python3 tools/check_dashes.py

Hyphens only, everywhere. The banned characters are built from their code
points on purpose, so that this file never contains one itself.
"""

import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

BANNED = {
    chr(0x2010): "non breaking hyphen",
    chr(0x2011): "non breaking hyphen",
    chr(0x2012): "figure dash",
    chr(0x2013): "en dash",
    chr(0x2014): "em dash",
    chr(0x2015): "horizontal bar",
    chr(0x2212): "minus sign",
}

SUFFIXES = {".ipynb", ".html", ".md", ".py", ".json", ".txt",
            ".R", ".r", ".css", ".js", ".yml", ".yaml"}

# Directories that are not ours to police: git internals, caches, and the
# .venv full of other people's packages that start.py installs.
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", ".ipynb_checkpoints",
             "node_modules"}


def ours(path):
    return not (SKIP_DIRS & set(path.parts))


def main():
    hits = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in SUFFIXES:
            continue
        if not ours(path):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for ch, name in BANNED.items():
                if ch in line:
                    hits.append("%s:%d  %s  %s"
                                % (path.relative_to(ROOT), lineno, name, line.strip()[:90]))
    # The Word file is binary, so read its text through pandoc when available.
    docx = ROOT / "reference" / "function_reference.docx"
    if docx.exists() and shutil.which("pandoc"):
        out = subprocess.run(["pandoc", "-t", "plain", str(docx)],
                             capture_output=True, text=True)
        if out.returncode == 0:
            for lineno, line in enumerate(out.stdout.splitlines(), start=1):
                for ch, name in BANNED.items():
                    if ch in line:
                        hits.append("reference/function_reference.docx:%d  %s  %s"
                                    % (lineno, name, line.strip()[:90]))

    if hits:
        print("Found %d banned dash characters:" % len(hits))
        for hit in hits:
            print("  " + hit)
        return 1
    n_files = sum(1 for p in ROOT.rglob("*")
                  if p.is_file() and p.suffix in SUFFIXES and ours(p))
    print("Scanned %d files. No em dashes, en dashes or lookalikes. Hyphens only." % n_files)
    return 0


if __name__ == "__main__":
    sys.exit(main())
