"""Build the test notebooks and revision sheets from tools/quiz_specs.

    python3 tools/build_quiz.py

For each spec it writes three things:

    quizzes/<slug>.ipynb                   the test, with the blanks in it
    solutions/quizzes/<slug>_solution.ipynb  the worked version, which must pass
    quizzes/<slug_without_quiz>_revision.md  the revision sheet

The test notebook scores itself with check(), which gives a mark out of the
number of questions, a subtotal per section, and a revision pointer for
everything missed, without revealing any answers. review() then prints the
worked review: every question, the model answer and why it is that one. The
review text is encoded rather than written into the notebook in plain sight,
for the same reason the expected answers are.
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

import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import build  # noqa: E402  the grading helpers and notebook metadata live there

SPEC_DIR = ROOT / "tools" / "quiz_specs"


QUIZ_GRADING = '''_META = _json.loads(_b64.b64decode("__META_BLOB__").decode("utf-8"))
_REVIEW = _json.loads(_b64.b64decode("__REVIEW_BLOB__").decode("utf-8"))


def check():
    """Score the test. Says what is wrong, never what the answer is."""
    global CHECK_RESULTS
    CHECK_RESULTS = []
    for item in _META["items"]:
        fn = globals().get("_part_" + item["id"])
        if fn is None:
            CHECK_RESULTS.append((item["id"], "FAIL", "no check exists for this question"))
            continue
        try:
            problem = fn()
            status = "FAIL" if problem else "PASS"
            msg = problem or ""
        except _Todo as exc:
            status, msg = "TODO", str(exc)
        except Exception as exc:
            status = "FAIL"
            msg = ("could not check this question, it raised %s: %s"
                   % (type(exc).__name__, exc))
        CHECK_RESULTS.append((item["id"], status, msg))
    _print_score()


def _by_id(item_id):
    for item in _META["items"]:
        if item["id"] == item_id:
            return item
    return {}


def _print_score():
    bar = "=" * 72
    print(bar)
    print("%s  |  %s" % (_META["quiz_id"], _META["title"]))
    print(bar)
    marks = 0
    for section in _META["sections"]:
        rows = [(i, s, m) for i, s, m in CHECK_RESULTS
                if _by_id(i).get("section") == section["key"]]
        got = sum(1 for _, s, _ in rows if s == "PASS")
        marks += got
        print("")
        print("Section %s, %s: %d of %d" % (section["key"], section["title"],
                                            got, len(rows)))
        for item_id, status, msg in rows:
            item = _by_id(item_id)
            print("  Question %s: %-4s  %s" % (item_id, status, item.get("topic", "")))
            if status != "PASS" and msg:
                for line in _tw.wrap(msg, width=62):
                    print("        " + line)
            if status != "PASS" and item.get("revise"):
                print("        revise: %s" % item["revise"])
    total = len(CHECK_RESULTS)
    print("")
    print("-" * 72)
    print("Total: %d of %d" % (marks, total))
    missed = [i for i, s, _ in CHECK_RESULTS if s != "PASS"]
    if missed:
        print("Not yet right: %s" % ", ".join(missed))
    print("Now run review(), whatever you scored.")
    print(bar)
    # The machine readable lines the verifier looks for.
    print("")
    for item_id, status, _ in CHECK_RESULTS:
        print("Question %s: %s" % (item_id, status))


def review():
    """The worked review: every question, the model answer, and why."""
    bar = "=" * 72
    print(bar)
    print("REVIEW  |  %s" % _META["title"])
    print(bar)
    print("")
    print("One way to do each question, and the reason it is that way. The checks")
    print("test the numbers and not the route, so a different correct answer is")
    print("still correct.")
    status_of = {i: s for i, s, _ in CHECK_RESULTS}
    for section in _META["sections"]:
        print("")
        print("=" * 72)
        print("SECTION %s, %s" % (section["key"], section["title"].upper()))
        print("=" * 72)
        for item in _META["items"]:
            if item["section"] != section["key"]:
                continue
            got = status_of.get(item["id"], "not scored")
            print("")
            print("-" * 72)
            print("Question %s  [%s]  %s" % (item["id"], got, item.get("topic", "")))
            print("-" * 72)
            entry = _REVIEW.get(item["id"], {})
            code = entry.get("code", "").strip()
            if code:
                print("")
                for line in code.splitlines():
                    print("    " + line)
            text = entry.get("text", "").strip()
            if text:
                print("")
                for para in text.split("\\n\\n"):
                    for line in _tw.wrap(" ".join(para.split()), width=70):
                        print(line)
                    print("")
    print(bar)
    if CHECK_RESULTS:
        marks = sum(1 for _, s, _ in CHECK_RESULTS if s == "PASS")
        print("You scored %d of %d. Revise what you missed, then move on."
              % (marks, len(CHECK_RESULTS)))
    else:
        print("check() has not been run, so there is no score to report.")
    print(bar)


def send_to_tutor():
    """Build the block you copy to your tutor, along with your code."""
    lines = []
    lines.append("===== SEND TO TUTOR =====")
    lines.append("Test       : %s, %s" % (_META["quiz_id"], _META["title"]))
    lines.append("Date       : %s" % _dt.date.today().isoformat())
    if not CHECK_RESULTS:
        lines.append("check() has not been run yet. Run the scoring cell above first.")
    else:
        marks = sum(1 for _, s, _ in CHECK_RESULTS if s == "PASS")
        lines.append("Score      : %d of %d" % (marks, len(CHECK_RESULTS)))
        for section in _META["sections"]:
            rows = [(i, s) for i, s, _ in CHECK_RESULTS
                    if _by_id(i).get("section") == section["key"]]
            got = sum(1 for _, s in rows if s == "PASS")
            lines.append("  section %s, %s: %d of %d"
                         % (section["key"], section["title"], got, len(rows)))
        lines.append("Questions not right:")
        any_missed = False
        for item_id, status, msg in CHECK_RESULTS:
            if status == "PASS":
                continue
            any_missed = True
            item = _by_id(item_id)
            lines.append("  %s (%s) %s" % (item_id, status, item.get("topic", "")))
            if msg:
                for line in _tw.wrap(msg, width=56):
                    lines.append("      " + line)
        if not any_missed:
            lines.append("  none")
    lines.append("=========================")
    lines.append("Paste the cells you want to talk about below this block.")
    return "\\n".join(lines)
'''


def load_specs():
    specs = []
    for path in sorted(SPEC_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        spec_obj = importlib.util.spec_from_file_location("quiz_" + path.stem, path)
        mod = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(mod)
        specs.append(mod)
    return specs


def encode(obj):
    text = json.dumps(build.json_safe(obj), separators=(",", ":"))
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def header_md(spec, is_solution):
    kind = "REFERENCE SOLUTION" if is_solution else "Test"
    n_items = len(spec.ITEMS)
    out = ["# %s: %s" % (kind, spec.TITLE), "",
           "| | |", "|---|---|",
           "| Test ID | `%s` |" % spec.QUIZ_ID,
           "| Covers | week%s %s |" % ("s" if len(spec.WEEKS) > 1 else "",
                                       " and ".join(str(w) for w in spec.WEEKS)),
           "| Questions | %d |" % n_items,
           "| Time | about %d minutes |" % spec.MINUTES, ""]
    if is_solution:
        out += ["> This is the worked version of the test. Only open it after you "
                "have sat the test and read `review()` in your own copy.", ""]
    else:
        out += [spec.INTRO_MD.strip(), ""]
    return "\n".join(out)


def build_one(spec):
    # 1. run the setup and every model answer, in order, to get the expected values
    ns = {"__name__": "__quiz_run__"}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(spec.DATA_CODE, "<data:%s>" % spec.QUIZ_ID, "exec"), ns)
        for item in spec.ITEMS:
            exec(compile(item["solution"], "<sol:%s%s>" % (spec.QUIZ_ID, item["id"]),
                         "exec"), ns)
        expected = spec.build_expected(ns)

    # Every question needs a check, a topic and somewhere to go back to.
    for item in spec.ITEMS:
        for field in ("id", "section", "topic", "revise", "prompt_md", "starter",
                      "solution", "review_md"):
            if not str(item.get(field, "")).strip():
                raise SystemExit("%s question %s: %s is empty"
                                 % (spec.QUIZ_ID, item.get("id"), field))
        if "_part_%s" % item["id"] not in spec.CHECK_CODE:
            raise SystemExit("%s question %s: no _part_%s in CHECK_CODE"
                             % (spec.QUIZ_ID, item["id"], item["id"]))
        if "# YOUR CODE" not in item["starter"]:
            raise SystemExit("%s question %s: the starter has no # YOUR CODE blank"
                             % (spec.QUIZ_ID, item["id"]))
    keys = [s["key"] for s in spec.SECTIONS]
    for item in spec.ITEMS:
        if item["section"] not in keys:
            raise SystemExit("%s question %s: unknown section %r"
                             % (spec.QUIZ_ID, item["id"], item["section"]))

    meta = {
        "quiz_id": spec.QUIZ_ID,
        "title": spec.TITLE,
        "sections": [{"key": s["key"], "title": s["title"]} for s in spec.SECTIONS],
        "items": [{"id": i["id"], "section": i["section"], "topic": i["topic"],
                   "revise": i["revise"]} for i in spec.ITEMS],
    }
    review = {i["id"]: {"code": i["solution"].strip(), "text": i["review_md"].strip()}
              for i in spec.ITEMS}

    grading = (build.GRADING_HELPERS + QUIZ_GRADING)
    grading = grading.replace("__EXPECTED_BLOB__", encode(expected))
    grading = grading.replace("__META_BLOB__", encode(meta))
    grading = grading.replace("__REVIEW_BLOB__", encode(review))
    check_code = grading + "\n\n" + spec.CHECK_CODE.strip() + "\n\n\ncheck()\n"

    written = []
    for is_solution in (False, True):
        cells = [new_markdown_cell(header_md(spec, is_solution))]
        cells.append(new_markdown_cell(
            "## Setup\n\nRun this cell. It builds the study every question works on. "
            "The data is synthetic and comes from a fixed seed, so your numbers are "
            "the same as mine and the same as the review's."))
        cells.append(new_code_cell(spec.DATA_CODE.strip() + "\n"))
        for section in spec.SECTIONS:
            cells.append(new_markdown_cell(
                "# Section %s. %s\n\n%s" % (section["key"], section["title"],
                                            section["intro"])))
            for item in spec.ITEMS:
                if item["section"] != section["key"]:
                    continue
                cells.append(new_markdown_cell(
                    "## Question %s\n%s" % (item["id"], item["prompt_md"].rstrip())))
                body = item["solution"] if is_solution else item["starter"]
                cells.append(new_code_cell(body.strip() + "\n"))
        cells.append(new_markdown_cell(
            "## Score it\n\nRun this cell. It marks every question, gives a subtotal "
            "per section, and points you at what to revise for anything you missed. "
            "It does not tell you the answers. Do not edit it."))
        cells.append(new_code_cell(check_code))
        cells.append(new_markdown_cell(
            "## The review\n\nRun this cell after you have scored. It walks through "
            "every question with a model answer and the reasoning, including the "
            "ones you got right, several of which pass for a reason that is not "
            "obvious."))
        cells.append(new_code_cell("review()\n"))
        cells.append(new_markdown_cell(
            "## Send to tutor\n\nRun this and copy what it prints, along with any "
            "cell you want to talk about."))
        cells.append(new_code_cell("print(send_to_tutor())\n"))

        nb = new_notebook(cells=cells, metadata=build.nb_metadata())
        prefix = spec.SLUG + ("-sol" if is_solution else "")
        build.stamp_ids(nb, prefix)
        rel = ("solutions/quizzes/%s_solution.ipynb" % spec.SLUG if is_solution
               else "quizzes/%s.ipynb" % spec.SLUG)
        out = ROOT / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, str(out))
        written.append(rel)

    rev_name = spec.SLUG.replace("_quiz", "") + "_revision.md"
    rev_path = ROOT / "quizzes" / rev_name
    rev_path.write_text(spec.REVISION_MD.strip() + "\n", encoding="utf-8")
    written.append("quizzes/" + rev_name)

    return written


def main():
    specs = load_specs()
    if not specs:
        print("no quiz specs found in %s" % SPEC_DIR)
        return 0
    for spec in specs:
        for rel in build_one(spec):
            size = (ROOT / rel).stat().st_size
            print("built %-12s %-46s %6.1f KB" % (spec.QUIZ_ID, rel, size / 1024.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
