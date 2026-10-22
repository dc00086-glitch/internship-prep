"""Build problem and solution notebooks from the spec modules in tools/specs.

You should not need to touch this file. Run it with:

    python3 tools/build.py

It regenerates everything under problems/ and solutions/, and refreshes the
problem list inside index.html.
"""

import base64
import contextlib
import importlib.util
import io
import json
import os
import pathlib
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPEC_DIR = ROOT / "tools" / "specs"

OWNER = "dc00086-glitch"
REPO = "internship-prep"
BRANCH = "main"


# --------------------------------------------------------------------------
# the checker preamble that every problem notebook carries
# --------------------------------------------------------------------------

CHECK_PREAMBLE = '''# Grading helpers. You do not need to read or edit this cell.
import base64 as _b64, json as _json, datetime as _dt, textwrap as _tw
import numpy as _np
import pandas as _pd

# Expected answers, encoded so that you do not read them by accident.
_EXP = _json.loads(_b64.b64decode("__EXPECTED_BLOB__").decode("utf-8"))

CHECK_RESULTS = []


class _Todo(Exception):
    """Raised when a part has not been attempted yet."""


def _need(name):
    g = globals()
    if name not in g:
        raise _Todo("`" + name + "` is not defined yet. Run the starter cell for this part.")
    val = g[name]
    if val is None:
        raise _Todo("`" + name + "` is still None. Fill in the # YOUR CODE blank.")
    return val


def _as_scalar(val):
    try:
        arr = _np.asarray(val, dtype=float)
    except Exception:
        return None
    if arr.size != 1:
        return None
    return float(arr.reshape(-1)[0])


def _as_array(val):
    try:
        return _np.asarray(val, dtype=float)
    except Exception:
        return None


def _num_close(a, b, rtol=1e-4, atol=1e-8):
    a = _as_scalar(a)
    if a is None:
        return False
    return bool(_np.isclose(a, float(b), rtol=rtol, atol=atol, equal_nan=True))


def _arr_close(a, b, rtol=1e-4, atol=1e-8):
    a = _as_array(a)
    if a is None:
        return False
    b = _np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return False
    return bool(_np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))


def _cols_of(obj):
    try:
        return [str(c) for c in obj.columns]
    except Exception:
        return None


def check():
    """Test every part and print PASS, FAIL or TODO for each one."""
    global CHECK_RESULTS
    names = sorted(n for n in globals() if n.startswith("_part_"))
    CHECK_RESULTS = []
    for name in names:
        label = name[len("_part_"):]
        try:
            problem = globals()[name]()
            status = "FAIL" if problem else "PASS"
            msg = problem or ""
        except _Todo as exc:
            status, msg = "TODO", str(exc)
        except Exception as exc:
            status = "FAIL"
            msg = "could not check this part, it raised %s: %s" % (type(exc).__name__, exc)
        CHECK_RESULTS.append((label, status, msg))
    _print_results()


def _print_results():
    bar = "=" * 70
    print(bar)
    print("Problem %s  |  %s" % (PROBLEM_ID, TOPIC))
    print(bar)
    for label, status, msg in CHECK_RESULTS:
        print("Part %s: %s" % (label, status))
        if msg:
            for line in _tw.wrap(msg, width=64):
                print("    " + line)
    n_pass = sum(1 for _, s, _ in CHECK_RESULTS if s == "PASS")
    print("-" * 70)
    print("Result: %d of %d parts passing" % (n_pass, len(CHECK_RESULTS)))
    if n_pass == len(CHECK_RESULTS):
        print("All parts pass. Run the last cell to build your tutor block.")
    print(bar)


def send_to_tutor():
    """Build the block you copy to your tutor, along with your code."""
    lines = []
    lines.append("===== SEND TO TUTOR =====")
    lines.append("Problem ID : %s" % PROBLEM_ID)
    lines.append("Topic      : %s" % TOPIC)
    lines.append("Date       : %s" % _dt.date.today().isoformat())
    lines.append("Check results:")
    if not CHECK_RESULTS:
        lines.append("  check() has not been run yet. Run the check cell above first.")
    else:
        for label, status, msg in CHECK_RESULTS:
            lines.append("  Part %s: %s" % (label, status))
            if msg:
                for line in _tw.wrap(msg, width=58):
                    lines.append("      " + line)
        n_pass = sum(1 for _, s, _ in CHECK_RESULTS if s == "PASS")
        lines.append("Summary: %d of %d parts passing" % (n_pass, len(CHECK_RESULTS)))
    lines.append("=========================")
    lines.append("Paste your code cells below this block.")
    return "\\n".join(lines)
'''

TUTOR_CELL = '''# Run this last. Copy everything it prints, then paste your own code under it.
print(send_to_tutor())
'''


# --------------------------------------------------------------------------
# turning python objects into something json can hold
# --------------------------------------------------------------------------

def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, pd.Series):
        return json_safe(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return {str(c): json_safe(obj[c].tolist()) for c in obj.columns}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    raise TypeError("cannot store %r of type %s in the expected blob" % (obj, type(obj)))


def encode_expected(expected):
    payload = json.dumps(json_safe(expected), sort_keys=True)
    return base64.b64encode(payload.encode("utf-8")).decode("ascii")


# --------------------------------------------------------------------------
# notebook assembly
# --------------------------------------------------------------------------

def colab_url(path):
    return "https://colab.research.google.com/github/%s/%s/blob/%s/%s" % (
        OWNER, REPO, BRANCH, path)


def nb_metadata():
    return {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    }


def header_md(spec, is_solution, nb_path):
    badge = "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](%s)" % colab_url(nb_path)
    kind = "REFERENCE SOLUTION" if is_solution else "Problem"
    out = ["# %s %s: %s" % (kind, spec.PID, spec.TITLE), "", badge, "",
           "| | |", "|---|---|",
           "| Problem ID | `%s` |" % spec.PID,
           "| Week / day | %d / %d |" % (spec.WEEK, spec.DAY),
           "| Topic | %s |" % spec.TOPIC,
           "| Expected time | %s |" % spec.EST, ""]
    if is_solution:
        out += ["> This is the worked solution. Only open it after you have "
                "genuinely tried the problem notebook.", ""]
    else:
        out += ["**How to work this notebook**", "",
                "1. Run the setup cell to build the data.",
                "2. Run the **Toolkit** cell. It demonstrates every move this "
                "problem needs, on a tiny made up example, so none of it is the "
                "answer. Read its output before you start typing.",
                "3. Fill in every `# YOUR CODE` blank, one part at a time.",
                "4. Run the check cell as often as you like. It tells you which "
                "part is wrong and why, without giving you the answer.",
                "5. When you are done, run the last cell and copy the tutor block.", "",
                "Stuck on something more basic, like what an array even is? The "
                "[foundations notebook](%s) covers Python, numpy, pandas and "
                "matplotlib from the ground up, and you can keep it open in a "
                "second tab." % colab_url("reference/foundations.ipynb"), "",
                "Nothing here needs installing. Everything used is already in Colab.", ""]
    return "\n".join(out)


def build_one(spec, mod_name):
    week_dir = "week%02d" % spec.WEEK
    prob_path = "problems/%s/%s.ipynb" % (week_dir, spec.SLUG)
    sol_path = "solutions/%s/%s_solution.ipynb" % (week_dir, spec.SLUG)

    # 1. run data + solution code to derive the expected answers
    ns = {"__name__": "__spec_run__"}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(spec.DATA_CODE, "<data:%s>" % spec.PID, "exec"), ns)
        for part in spec.PARTS:
            exec(compile(part["solution"], "<sol:%s%s>" % (spec.PID, part["label"]), "exec"), ns)
        expected = spec.build_expected(ns)

    # A toolkit runs after the setup cell, so it must not rebind anything the
    # setup cell made, or the student's data quietly changes under them.
    # Each part must be its own H2 section. At H3 it nests under the Toolkit
    # section in Colab, which hides the questions when that section collapses.
    for part in spec.PARTS:
        md = part["prompt_md"].strip()
        heads = [ln for ln in md.splitlines() if ln.startswith("#")]
        if not heads:
            raise SystemExit("%s part %s: prompt_md has no heading."
                             % (spec.PID, part["label"]))
        level = len(heads[0]) - len(heads[0].lstrip("#"))
        if level != 2:
            raise SystemExit(
                "%s part %s: prompt_md heading is H%d, it must be H2 so the part is a "
                "sibling of Setup and Toolkit rather than nested inside them."
                % (spec.PID, part["label"], level))
        if len(md.split()) < 40:
            raise SystemExit(
                "%s part %s: prompt_md is only %d words, which means it is a bare "
                "heading. Each part needs its own explanation above its cell."
                % (spec.PID, part["label"], len(md.split())))

    if getattr(spec, "TOOLKIT_CODE", None):
        data_ns = {"__name__": "__data_only__"}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(spec.DATA_CODE, "<data:%s>" % spec.PID, "exec"), data_ns)
            data_names = {k: v for k, v in data_ns.items() if not k.startswith("__")}
            tk_ns = dict(data_ns)
            exec(compile(spec.TOOLKIT_CODE, "<toolkit:%s>" % spec.PID, "exec"), tk_ns)
        clobbered = sorted(k for k, v in data_names.items()
                           if k in tk_ns and tk_ns[k] is not v)
        if clobbered:
            raise SystemExit(
                "%s: the toolkit cell rebinds %s, which the setup cell created. "
                "Rename the toolkit variable, usually with a demo_ prefix."
                % (spec.PID, ", ".join(clobbered)))
    blob = encode_expected(expected)

    check_code = CHECK_PREAMBLE.replace("__EXPECTED_BLOB__", blob) + "\n\n" + spec.CHECK_CODE.rstrip() + "\n\n\ncheck()\n"

    for is_solution, out_path in ((False, prob_path), (True, sol_path)):
        cells = [new_markdown_cell(header_md(spec, is_solution, out_path)),
                 new_markdown_cell(spec.SCENARIO_MD.strip())]
        if getattr(spec, "CONCEPT_MD", None):
            cells.append(new_markdown_cell(spec.CONCEPT_MD.strip()))
        cells += [new_markdown_cell(spec.TASK_MD.strip()),
                 new_markdown_cell("## Setup\n\nRun this cell first. It builds the "
                                   "data with a fixed random seed, so your numbers "
                                   "match the checker and match mine."),
                 new_code_cell(spec.DATA_CODE.strip() + "\n")]
        if getattr(spec, "TOOLKIT_MD", None):
            cells.append(new_markdown_cell(
                spec.TOOLKIT_MD.strip() +
                "\n\nRun the cell below, then carry on to **Part a**, which is the "
                "next section down."))
        if getattr(spec, "TOOLKIT_CODE", None):
            cells.append(new_code_cell(spec.TOOLKIT_CODE.strip() + "\n"))
        for part in spec.PARTS:
            cells.append(new_markdown_cell(part["prompt_md"].strip()))
            body = part["solution"] if is_solution else part["starter"]
            cells.append(new_code_cell(body.strip() + "\n"))
        cells.append(new_markdown_cell("## Check your work\n\nRun this cell. It "
                                       "prints PASS, FAIL or TODO for each part. "
                                       "Do not edit it."))
        cells.append(new_code_cell(check_code.strip() + "\n"))
        cells.append(new_markdown_cell("## Send to tutor"))
        cells.append(new_code_cell(TUTOR_CELL.strip() + "\n"))

        nb = new_notebook(cells=cells, metadata=nb_metadata())
        full = ROOT / out_path
        full.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, str(full))

    return {"id": spec.PID, "week": spec.WEEK, "day": spec.DAY,
            "title": spec.TITLE, "topic": spec.TOPIC, "est": spec.EST,
            "path": prob_path, "solution": sol_path}


def load_specs():
    specs = []
    for path in sorted(SPEC_DIR.glob("w*.py")):
        name = path.stem
        spec_obj = importlib.util.spec_from_file_location("spec_" + name, path)
        mod = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(mod)
        specs.append((mod, name))
    specs.sort(key=lambda pair: (pair[0].WEEK, pair[0].DAY))
    return specs


def patch_index(entries):
    index = ROOT / "index.html"
    if not index.exists():
        print("index.html not found, skipping")
        return
    text = index.read_text(encoding="utf-8")
    start = "// === PROBLEMS START ==="
    end = "// === PROBLEMS END ==="
    if start not in text or end not in text:
        print("index.html has no PROBLEMS markers, skipping")
        return
    body = json.dumps(entries, indent=2)
    block = "%s\n  const PROBLEMS = %s;\n  %s" % (start, body, end)
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    index.write_text(head + block + tail, encoding="utf-8")
    print("index.html problem list refreshed (%d problems)" % len(entries))


def main():
    specs = load_specs()
    if not specs:
        print("no specs found in %s" % SPEC_DIR)
        return 1
    entries = []
    for mod, name in specs:
        entry = build_one(mod, name)
        entries.append(entry)
        print("built %s  %s" % (entry["id"], entry["path"]))
    patch_index(entries)
    (ROOT / "problems.json").write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print("wrote problems.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
