"""Run every notebook top to bottom in a clean kernel and report the result.

    python3 tools/verify.py

Solution notebooks must end with every part passing.
Problem notebooks must execute without error, with every part TODO.
"""

import os
import pathlib
import re
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

import nbformat
from nbclient import NotebookClient

ROOT = pathlib.Path(__file__).resolve().parent.parent


def run_notebook(path):
    nb = nbformat.read(str(path), as_version=4)
    client = NotebookClient(nb, timeout=300, kernel_name="python3",
                            resources={"metadata": {"path": str(path.parent)}})
    client.execute()
    text = []
    for cell in nb.cells:
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                text.append(out.get("text", ""))
            elif out.get("output_type") == "execute_result":
                text.append(out.get("data", {}).get("text/plain", ""))
    return "\n".join(text)


def parse_results(text):
    return re.findall(r"^Part ([a-z]): (PASS|FAIL|TODO)$", text, flags=re.M)


def main():
    # Optional filter: python3 tools/verify.py week07
    only = sys.argv[1] if len(sys.argv) > 1 else None
    failures = []
    for kind in ("solutions", "problems", "reference"):
        if not (ROOT / kind).exists():
            continue
        for path in sorted((ROOT / kind).rglob("*.ipynb")):
            rel = path.relative_to(ROOT)
            if only and only not in str(rel):
                continue
            try:
                text = run_notebook(path)
            except Exception as exc:
                failures.append("%s: did not execute: %s: %s" % (rel, type(exc).__name__, str(exc)[:400]))
                print("EXEC FAIL  %s" % rel)
                continue

            if kind == "reference":
                print("ok         %s  (reference, ran clean)" % rel)
                continue

            results = parse_results(text)
            if not results:
                failures.append("%s: check() printed no part results" % rel)
                print("NO CHECK   %s" % rel)
                continue

            statuses = [s for _, s in results]
            if kind == "solutions":
                bad = [p for p, s in results if s != "PASS"]
                if bad:
                    failures.append("%s: parts not passing: %s" % (rel, ", ".join(bad)))
                    print("CHECK FAIL %s  (%s)" % (rel, ",".join(statuses)))
                else:
                    print("ok         %s  (%d parts PASS)" % (rel, len(results)))
            else:
                bad = [p for p, s in results if s != "TODO"]
                if bad:
                    failures.append("%s: blank notebook should report TODO, got other "
                                    "statuses for: %s" % (rel, ", ".join(bad)))
                    print("STATE FAIL %s  (%s)" % (rel, ",".join(statuses)))
                else:
                    print("ok         %s  (%d parts TODO)" % (rel, len(results)))

            if "SEND TO TUTOR" not in text:
                failures.append("%s: tutor block did not print" % rel)

    print("")
    if failures:
        print("FAILURES (%d):" % len(failures))
        for f in failures:
            print("  - " + f)
        return 1
    scope = ("matching %r" % only) if only else "All"
    print("%s notebooks execute cleanly. All solutions pass their own check()." % scope)
    return 0


if __name__ == "__main__":
    sys.exit(main())
