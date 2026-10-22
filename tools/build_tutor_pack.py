"""Build the files to upload into a Claude project, so the tutor knows the problems.

    python3 tools/build_tutor_pack.py

Writes into tutor/:

    TUTOR_INSTRUCTIONS.md     paste this into the project's custom instructions
    tutor_index.md            all 40 problems in one page, compact
    tutor_reference.md        every function, the error table, the house rules
    tutor_environment.md      Colab and offline setup, and what breaks
    tutor_weekNN.md           one file per week: the full text of each problem,
                              the data, the toolkit, the parts, and what every
                              check failure means
    tutor_weekNN_solutions.md the reference solutions for that week, kept in a
                              separate file so uploading them is a choice

Everything is generated from tools/specs/, the same source the notebooks are
built from, so the tutor can never be told something the notebooks do not do.
Rerun it whenever a problem changes.
"""

import ast
import importlib.util
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPEC_DIR = ROOT / "tools" / "specs"
OUT = ROOT / "tutor"

WEEK_TITLES = {
    1: "numpy, pandas and matplotlib",
    2: "pandas in depth",
    3: "scipy",
    4: "scikit-image and OpenCV",
    5: "scikit-learn",
    6: "PyTorch",
    7: "PK and PD applications, part 1",
    8: "PK and PD applications, part 2",
}


def load_specs():
    out = []
    for path in sorted(SPEC_DIR.glob("w*.py")):
        spec_obj = importlib.util.spec_from_file_location("tutor_" + path.stem, path)
        mod = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(mod)
        out.append(mod)
    out.sort(key=lambda m: (m.WEEK, m.DAY))
    return out


def blanks(starter):
    """The variable names a part asks for, in order, from its # YOUR CODE lines."""
    names = []
    for line in starter.splitlines():
        if "# YOUR CODE" not in line:
            continue
        m = re.match(r"\s*([A-Za-z_][\w]*)\s*(?:=|,)", line)
        if m and m.group(1) not in names:
            names.append(m.group(1))
        else:
            m2 = re.match(r"\s*([A-Za-z_][\w]*)\s*,\s*([A-Za-z_][\w]*)", line)
            if m2:
                for g in m2.groups():
                    if g not in names:
                        names.append(g)
    return names


def collect_strings(node, out):
    """String constants in message position, skipping dictionary keys.

    A message like `"holds %d of %d" % (n, _EXP["n_measurable"])` has strings
    in two different roles, and only the first is prose. Not descending into a
    subscript's key keeps "n_measurable" out of the text.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            out.append(node.value)
        return
    if isinstance(node, ast.Subscript):
        collect_strings(node.value, out)
        return
    for child in ast.iter_child_nodes(node):
        collect_strings(child, out)


def flatten_string(node):
    """Best effort text of a return value that may be a concatenation or a %."""
    parts = []
    collect_strings(node, parts)
    text = " ".join(p.strip() for p in parts if p.strip())
    return re.sub(r"\s+", " ", text).strip()


def check_messages(check_code):
    """{part label: [diagnostic text, ...]} pulled straight out of the check cell."""
    out = {}
    try:
        tree = ast.parse(check_code)
    except SyntaxError:
        return out
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("_part_"):
            continue
        label = node.name[len("_part_"):]
        msgs = []
        for sub in ast.walk(node):
            if isinstance(sub, ast.Return) and sub.value is not None:
                text = flatten_string(sub.value)
                if text and text.lower() != "none":
                    msgs.append(text)
        seen = set()
        out[label] = [m for m in msgs if not (m in seen or seen.add(m))]
    return out


def fence(code, lang="python"):
    return "```%s\n%s\n```" % (lang, code.strip("\n"))


# ----------------------------------------------------------------- the index

def build_index(specs):
    L = ["# Problem index",
         "",
         "Every problem in the curriculum, one line each, with the variables each",
         "part asks for. Use this to work out which problem the student is on and",
         "what it wants. The full text of a week is in tutor_weekNN.md.",
         ""]
    for week in sorted({m.WEEK for m in specs}):
        L += ["## Week %d, %s" % (week, WEEK_TITLES.get(week, "")), ""]
        for m in [s for s in specs if s.WEEK == week]:
            L += ["### %s. %s" % (m.PID, m.TITLE),
                  "",
                  "- topic: %s" % m.TOPIC,
                  "- notebook: problems/week%02d/%s.ipynb" % (m.WEEK, m.SLUG),
                  "- time: %s" % m.EST]
            for part in m.PARTS:
                L.append("- part %s asks for: %s"
                         % (part["label"], ", ".join("`%s`" % v for v in blanks(part["starter"])) or "see the notebook"))
            if getattr(m, "FUNCTIONS", None):
                L.append("- introduces: %s" % ", ".join(m.FUNCTIONS))
            L.append("")
    return "\n".join(L)


# ------------------------------------------------------------ per week detail

def build_week(specs, week):
    mods = [m for m in specs if m.WEEK == week]
    L = ["# Week %d, %s" % (week, WEEK_TITLES.get(week, "")),
         "",
         "The full text of every problem this week, the data it generates, the",
         "toolkit the student is given, what each part asks for, and what every",
         "possible check failure means. No solutions in this file.",
         ""]
    for m in mods:
        msgs = check_messages(m.CHECK_CODE)
        L += ["---", "",
              "# %s. %s" % (m.PID, m.TITLE),
              "",
              "- topic: %s" % m.TOPIC,
              "- notebook: problems/week%02d/%s.ipynb" % (m.WEEK, m.SLUG),
              "- solution: solutions/week%02d/%s_solution.ipynb" % (m.WEEK, m.SLUG),
              "- expected time: %s" % m.EST,
              ""]
        L += [m.SCENARIO_MD.strip(), ""]
        L += [m.CONCEPT_MD.strip(), ""]
        L += [m.TASK_MD.strip(), ""]
        L += ["## The setup cell the student runs", "",
              "This is given to them, already working. The data is synthetic and",
              "seeded, so their numbers are always these numbers.", "",
              fence(m.DATA_CODE), ""]
        L += [m.TOOLKIT_MD.strip(), "",
              "The toolkit cell demonstrates each of those on a tiny made up example.",
              "It is in the notebook and none of it is the answer.", ""]
        for part in m.PARTS:
            L += [part["prompt_md"].strip(), "",
                  "**Variables this part is graded on:** %s"
                  % (", ".join("`%s`" % v for v in blanks(part["starter"])) or "see the starter"),
                  "",
                  "The starter cell they see:", "",
                  fence(part["starter"]), ""]
            got = msgs.get(part["label"], [])
            if got:
                L += ["**Every failure message part %s can produce.** Each one is a real"
                      % part["label"],
                      "mistake the check was written to catch, so when the student quotes",
                      "one, it tells you exactly what they did. A `%s` or `%d` in a message",
                      "is filled in at the time with the variable name or the number they",
                      "got:", ""]
                L += ["- " + t for t in got]
                L.append("")
        if getattr(m, "FUNCTIONS", None):
            L += ["**Functions this problem introduces:** %s" % ", ".join(m.FUNCTIONS), ""]
    return "\n".join(L)


def build_week_solutions(specs, week):
    mods = [m for m in specs if m.WEEK == week]
    L = ["# Week %d reference solutions" % week,
         "",
         "One way to do each part. The checks test the numbers, not the route, so a",
         "different correct answer is still correct.",
         "",
         "Do not paste these to the student. Use them to work out where their",
         "version diverges, and say the smallest thing that gets them unstuck.",
         ""]
    for m in mods:
        L += ["---", "", "## %s. %s" % (m.PID, m.TITLE), ""]
        for part in m.PARTS:
            L += ["### Part %s" % part["label"], "", fence(part["solution"]), ""]
    return "\n".join(L)


# -------------------------------------------------------------- the reference

def build_reference():
    sys.path.insert(0, str(ROOT / "tools"))
    import function_catalog as cat

    L = ["# Function reference, error messages and house rules",
         "",
         "The same content as reference/function_reference.docx, which the student",
         "also has. Quote the watch out line when it is the cause of their problem.",
         ""]
    groups = []
    for g, *_ in cat.ENTRIES:
        if g not in groups:
            groups.append(g)
    for group in groups:
        L += ["## %s" % group, ""]
        for g, key, sig, what, example, watch in cat.ENTRIES:
            if g != group:
                continue
            L += ["### %s" % key,
                  "",
                  "- signature: `%s`" % sig,
                  "- what it does: %s" % what,
                  "- example: `%s`" % example.replace("\n", " "),
                  "- watch out: %s" % watch,
                  ""]
    L += ["## Error messages and what they really mean", ""]
    for err, meaning in cat.ERRORS:
        L += ["### `%s`" % err, "", meaning, ""]
    L += ["## House rules, enforced by the checks", ""]
    for rule, why in cat.RULES:
        L += ["### %s" % rule, "", why, ""]
    return "\n".join(L)


# ------------------------------------------------------------- environment

ENVIRONMENT = """# How the student runs these, and what breaks

## Two routes, same notebooks

**Colab.** They open https://dc00086-glitch.github.io/internship-prep/, press
the Colab link on today's problem, and work in the browser. Nothing is
installed. Their edits live in their own Colab copy, not in the repository.

**Their own computer.** `python start.py` in the project folder installs
numpy, pandas, matplotlib, scipy, scikit-learn, scikit-image, OpenCV,
JupyterLab and PyTorch into a `.venv` inside the folder, then opens today's
problem in JupyterLab. The first run needs internet; every run after is
offline. `python start.py 12` opens a particular problem, `--list` prints the
schedule, `--browse` opens the one page view, `--foundations` opens the
foundations notebook, `--no-torch` skips the one large download, `--no-venv`
uses an existing Anaconda environment.

Either way the notebook files are identical, so advice never has to differ.

## How a problem notebook is laid out

| Cell | What it is |
|---|---|
| header | problem ID, topic, how to work it |
| Scenario | the pharmacology framing |
| How it works | how the package idea works, written out properly |
| Your task | parts a, b, c, with the exact variable names wanted |
| Setup | synthetic data from a fixed seed, no uploads |
| Toolkit | every move demonstrated on a tiny made up example |
| Part a, b, c | `# YOUR CODE` blanks, each with its own explanation |
| Check | `check()`, printing PASS, FAIL with a reason, or TODO, per part |
| Send to tutor | the block they paste into chat |

## How check() works, and what its answers mean

- **PASS** means the numbers are right. It does not mean the code is tidy.
- **FAIL** means a specific wrong thing was detected, and the message says
  which. The messages are written to diagnose without revealing, so a FAIL
  never contains the answer. Each week file lists every message a part can
  produce, which is the fastest way to know what the student actually did.
- **TODO** means the variable is still `None` or undefined, so they have not
  started that part or did not run the cell.
- An **unexpected exception** inside a part is reported as a crash with the
  exception text, which usually means a shape or type error upstream.

The expected answers are stored in the check cell as a base64 blob so they
cannot be read by accident. Tell the student not to decode it; it will not
teach them anything and it removes the only feedback loop they have.

## The schedule

One problem per weekday from the start date in `index.html`, currently
2026-10-19. Weekends hold on Friday's problem. `?n=3` on the site, or
`python start.py 3`, opens any problem out of order.

## Things that actually go wrong

**`NameError`.** Cells run out of order. The fix is Runtime, then Restart and
run all, in Colab, or Kernel, then Restart kernel and run all cells, in
JupyterLab. This is the single most common cause of confusing behaviour.

**A part says TODO when they think they answered it.** They edited the cell
and did not run it, or they ran the check cell before the part cell.

**`np.std` giving a smaller number than expected.** numpy divides by n unless
you pass `ddof=1`. Pandas `.std()` already divides by `n - 1`. Several checks
test for exactly this.

**SEM where SD was asked for.** The checks name it. SEM shrinks with sample
size and answers a different question.

**Shapes.** `axis=0` goes down the columns and `axis=1` across the rows, and
with a subjects by times matrix both are plausible numbers, so the mistake
does not raise. Tell them to print `.shape` before anything else.

**Matplotlib figures not appearing.** In JupyterLab they need nothing special,
`plt.show()` works. If a plot cell is slow or blank, it is usually that an
earlier cell raised.

**`No module named cv2` or similar, local only.** The install did not finish.
`python start.py --reinstall`.

**PyTorch missing, local only.** Only week 6 needs it. It is the one large
download and the launcher may have skipped it, which it says at the time.

**A solver result that moves when tolerances change.** Several problems ask for
`rtol=1e-8, atol=1e-10` or tighter and the checks are tight enough to notice,
because default tolerances leave error an optimiser will chase.

**Version differences.** The notebooks avoid APIs that moved: no
`squared=False`, no `np.trapz`, no `min_size=` on `remove_small_objects`. They
are verified on two package stacks, one Colab-like and one current. If
something still looks version dependent, it is worth reporting rather than
working around.

## Rebuilding, if they ever need a blank notebook back

`python3 tools/build.py` rewrites every problem notebook from its spec, which
restores the blanks and discards whatever they had typed. Their answers should
be copied out first. Problem and solution notebooks are generated from the same
spec, so a solution cannot drift from its problem.
"""


# ------------------------------------------------------------- instructions

INSTRUCTIONS = """# Tutor instructions

Paste this whole file into the project's custom instructions. Upload the other
files in this folder as project knowledge.

---

You are the tutor for a problem of the day curriculum in clinical pharmacology
and pharmacometrics. The student is a pharmacology PhD student preparing for an
industry internship. They are an intermediate Python and R user and not a
software engineer, and they are doing this to be able to hold their own in a
modelling group.

## What you have

- **tutor_index.md**: all 40 problems, with the variable names each part asks
  for. Start here to work out which problem they are on.
- **tutor_weekNN.md**: the full text of that week's problems, the setup code,
  the toolkit, every part prompt, and every failure message each check can
  produce. The failure messages are the fast path: when the student quotes one,
  that file tells you exactly what they did wrong.
- **tutor_reference.md**: every function the curriculum uses, with the mistake
  each one invites, plus the error message table and the house rules.
- **tutor_environment.md**: how they run the notebooks, how `check()` behaves,
  and the things that actually go wrong.
- **tutor_weekNN_solutions.md**, if it was uploaded: one correct version of
  each part.

If the problem they mention is not in the files you have, say so and ask them
to paste the part prompt. Do not guess at what a problem asks for.

## How a conversation starts

They paste a **Send to tutor** block, which carries the problem ID, the topic
and the check results, usually with their code. Read the check results first.
`PASS` means that part is numerically right. `FAIL` comes with a diagnostic you
can look up. `TODO` means they have not attempted it or did not run the cell.

If they paste nothing but an error message, ask for the problem ID and the
cell. If they paste code with no question, assume they want to know why the
check is failing.

## How to help

**Diagnose before explaining.** Find the specific line that produces the wrong
number. One sentence naming the cause is worth more than a paragraph of
background.

**Give the smallest hint that unblocks them.** The first answer should point at
the right line or the right concept, not contain working code. If they ask
again, go further. If they ask you outright for the answer, give it, and then
explain what it was about the problem that made it hard to see.

**Never paste a whole solution cell unasked**, even when you have the solutions
file. The checks exist so they get feedback without being handed answers, and
you undo the whole curriculum by short circuiting that.

**Use their variable names.** Every part names exactly what it wants. Say
`c_measurable`, not "your filtered array".

**Write runnable fragments, not pseudocode.** One or two lines, in the style
the notebook already uses. No type hints, no classes they did not ask for, no
refactoring of code that works.

**Explain the pharmacology as well as the code.** Why clearance comes from
AUC to infinity rather than AUC last, why the nadir of an indirect response
moves with dose, why a pooled fit biases the typical value. That reasoning is
the point of the internship; the syntax is the vehicle.

**Correct the premise when it is wrong.** If they think a 20 percent dose
proportionality deviation in eight subjects means saturable metabolism, say
plainly that it does not and why. Agreeing with a wrong reading to be
encouraging is the one failure mode that actually costs them in a meeting.

## House rules you are expected to enforce

- **Error bars are SD, never SEM.** SEM reports how precisely the mean was
  located and shrinks as n grows. SD reports how much subjects differ, which is
  the question being asked.
- **SD means the sample SD.** `np.std` divides by n unless given `ddof=1`.
  Pandas `.std()` already divides by `n - 1`.
- **Nested data is aggregated to the animal, or the subject, first.** Many
  cells inside one animal are repeat measures of that animal. Pooling them
  inflates n and weights each animal by however many cells were imaged.
- **Check `len()` before and after every merge.** A join that silently drops a
  subject produces a plausible answer and no error.
- **Geometric means and CVs for PK parameters**, because they are lognormal,
  and percentiles rather than mean plus or minus SD for skewed distributions.
- **Look at the parameter estimates, not only the loss.** A fit whose loss
  stopped moving can be sitting on a volume of distribution of fourteen
  million litres.

## Style

No em dashes or en dashes, anywhere, ever. Hyphens only. The whole curriculum
is written that way and the student checks for it.

Short paragraphs. No bullet lists of options when one recommendation will do.
No praise at the start of a message. No restating their question back to them.
If a number matters, give the number.

When you are not sure of a numerical claim, say so, or work it out rather than
asserting it. The notebooks are built so that every number in them was read off
real output, and the tutor should hold itself to the same standard.

## What not to do

Do not tell them to decode the base64 expected answers in the check cell. Do
not suggest pip installing things in Colab; everything they need is already
there. Do not rewrite a working cell to be more idiomatic. Do not introduce
pathlib, dataclasses, type hints or comprehensions where a loop is clearer, and
do not suggest a package the curriculum does not use.
"""


def main():
    specs = load_specs()
    OUT.mkdir(exist_ok=True)
    written = []

    def write(name, text):
        path = OUT / name
        path.write_text(text.rstrip() + "\n", encoding="utf-8")
        written.append((name, path.stat().st_size))

    write("TUTOR_INSTRUCTIONS.md", INSTRUCTIONS)
    write("tutor_index.md", build_index(specs))
    write("tutor_reference.md", build_reference())
    write("tutor_environment.md", ENVIRONMENT)
    for week in sorted({m.WEEK for m in specs}):
        write("tutor_week%02d.md" % week, build_week(specs, week))
        write("tutor_week%02d_solutions.md" % week, build_week_solutions(specs, week))

    print("wrote %d files into %s" % (len(written), OUT))
    total = 0
    for name, size in written:
        total += size
        print("  %-34s %7.1f KB  about %5d words" % (name, size / 1024.0, size // 6))
    print("  %-34s %7.1f KB" % ("TOTAL", total / 1024.0))
    print("")
    print("Upload to the project: TUTOR_INSTRUCTIONS.md as the custom")
    print("instructions, then tutor_index.md, tutor_reference.md,")
    print("tutor_environment.md, and the week files you are working on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
