"""Build reference/function_reference.docx from the catalog and the specs.

    python3 tools/build_glossary.py

Every function in tools/function_catalog.py is listed, tagged with the problem
that introduces it. Anything no problem claims is marked as coming from the
foundations notebook. Add a week, list its functions in the new specs, and
this document grows to match.
"""

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import datetime

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPEC_DIR = ROOT / "tools" / "specs"
OUT = ROOT / "reference" / "function_reference.docx"
GROUP_ORDER = ["Python", "numpy", "pandas", "scipy", "matplotlib", "imaging", "sklearn",
               "torch", "PK"]


def load(path, name):
    spec_obj = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(mod)
    return mod


def main():
    catalog = load(ROOT / "tools" / "function_catalog.py", "catalog")

    specs = []
    for path in sorted(SPEC_DIR.glob("w*.py")):
        specs.append(load(path, "spec_" + path.stem))
    specs.sort(key=lambda m: (m.WEEK, m.DAY))

    known = {entry[1] for entry in catalog.ENTRIES}

    # which problem introduces each function
    first_seen = {}
    problems = []
    for spec in specs:
        keys = list(getattr(spec, "FUNCTIONS", []))
        unknown = [k for k in keys if k not in known]
        if unknown:
            raise SystemExit(
                "%s lists %s, which is not in tools/function_catalog.py. Add an "
                "entry there first." % (spec.PID, ", ".join(repr(u) for u in unknown)))
        fresh = []
        for key in keys:
            if key not in first_seen:
                first_seen[key] = spec.PID
                fresh.append(key)
        problems.append({"pid": spec.PID, "week": spec.WEEK, "day": spec.DAY,
                         "title": spec.TITLE, "topic": spec.TOPIC,
                         "new": fresh, "n_new": len(fresh)})

    groups = []
    for group in GROUP_ORDER:
        rows = []
        for g, key, sig, what, example, watch in catalog.ENTRIES:
            if g != group:
                continue
            rows.append({"key": key, "sig": sig, "what": what,
                         "example": example, "watch": watch,
                         "seen": first_seen.get(key, "Foundations")})
        if rows:
            groups.append({"name": group, "rows": rows})

    weeks = sorted({s.WEEK for s in specs})
    payload = {
        "title": "Function Reference",
        "subtitle": "Every function used in the internship-prep problems, "
                    "with what it does and what to watch for.",
        "generated": datetime.date.today().isoformat(),
        "weeks_built": weeks,
        "weeks_label": ", ".join(str(w) for w in weeks),
        "n_functions": len(catalog.ENTRIES),
        "n_problems": len(problems),
        "repo": "github.com/dc00086-glitch/internship-prep",
        "problems": problems,
        "groups": groups,
        "errors": [{"msg": m, "means": v} for m, v in catalog.ERRORS],
        "rules": [{"rule": r, "why": w} for r, w in catalog.RULES],
    }

    tmp = ROOT / "tools" / ".glossary_data.json"
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["node", str(ROOT / "tools" / "make_docx.js"), str(tmp), str(OUT)],
        capture_output=True, text=True)
    tmp.unlink(missing_ok=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit("make_docx.js failed")

    print(result.stdout.strip())
    n_found = sum(len(g["rows"]) for g in groups)
    n_claimed = len(first_seen)
    print("%d functions, %d introduced by a problem, %d from the foundations notebook"
          % (n_found, n_claimed, n_found - n_claimed))
    print("covers week(s): %s" % payload["weeks_label"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
