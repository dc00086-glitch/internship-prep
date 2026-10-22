#!/usr/bin/env python3
"""Open today's problem on this computer, with no Colab and no internet.

Usage, from a terminal in this folder:

    python start.py              today's problem, in JupyterLab
    python start.py 12           problem 12 instead
    python start.py --list       the whole schedule, with dates
    python start.py --browse     open the browse page in a web browser
    python start.py --foundations open the foundations notebook
    python start.py --setup      install everything and stop
    python start.py --no-torch   skip the PyTorch download (week 6 needs it)

The first run installs the packages into a folder called .venv inside this
one, which needs internet and takes a few minutes. Every run after that is
offline and starts in a couple of seconds.

Nothing is installed into your system Python, and deleting the .venv folder
undoes everything this script did.
"""

import hashlib
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import webbrowser
from datetime import date, timedelta

ROOT = pathlib.Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQS = ROOT / "requirements.txt"
REQS_TORCH = ROOT / "requirements-torch.txt"
STAMP = VENV / "installed.json"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
MIN_PYTHON = (3, 9)


# --------------------------------------------------------------- small helpers

def say(msg=""):
    print(msg, flush=True)


def step(msg):
    say("")
    say("-> " + msg)


def die(msg, *extra):
    say("")
    say("STOPPED: " + msg)
    for line in extra:
        say("  " + line)
    sys.exit(1)


def venv_python():
    if platform.system() == "Windows":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def in_our_venv():
    try:
        return pathlib.Path(sys.prefix).resolve() == VENV.resolve()
    except OSError:
        return False


def run(args, **kw):
    return subprocess.call([str(a) for a in args], cwd=str(ROOT), **kw)


# ------------------------------------------------------------------ the venv

def make_venv():
    if venv_python().exists():
        return
    step("Making a private package folder at .venv (one off)")
    code = run([sys.executable, "-m", "venv", str(VENV)])
    if code != 0 or not venv_python().exists():
        hint = "sudo apt install python3-venv" if platform.system() == "Linux" else ""
        die("could not create the .venv folder.",
            "Your Python is missing the venv module." + ((" Try: " + hint) if hint else ""),
            "Or run everything in your current Python instead:",
            "    python start.py --no-venv")


def reexec_in_venv(argv):
    py = venv_python()
    step("Switching to the .venv Python")
    sys.exit(run([py, str(ROOT / "start.py"), "--in-venv"] + argv))


# --------------------------------------------------------------- installing

def req_hash(with_torch):
    h = hashlib.sha256()
    h.update(REQS.read_bytes())
    if with_torch:
        h.update(REQS_TORCH.read_bytes())
    h.update(b"torch" if with_torch else b"no-torch")
    return h.hexdigest()[:16]


def already_installed(with_torch):
    if not STAMP.exists():
        return False
    try:
        info = json.loads(STAMP.read_text())
    except (ValueError, OSError):
        return False
    return info.get("hash") == req_hash(with_torch)


def pip_install(args, label):
    step(label)
    code = run([sys.executable, "-m", "pip", "install"] + list(args))
    if code != 0:
        die("the install failed.",
            "The usual cause is no internet connection, since the first run has",
            "to download the packages once. Check the connection and run it again.",
            "If you are behind a proxy that blocks pip, install these yourself:",
            "    " + " ".join(REQS.read_text().split()[-8:]))


def install_packages(with_torch, force=False):
    if already_installed(with_torch) and not force:
        say("Packages are already installed. Skipping the install.")
        return
    pip_install(["--upgrade", "pip"], "Updating pip")
    pip_install(["-r", str(REQS)], "Installing numpy, pandas, matplotlib, scipy, "
                                   "scikit-learn, scikit-image, OpenCV and JupyterLab")
    if with_torch:
        pip_install(["-r", str(REQS_TORCH), "--index-url", TORCH_CPU_INDEX],
                    "Installing PyTorch, CPU build, used by week 6")
    STAMP.write_text(json.dumps({"hash": req_hash(with_torch),
                                 "torch": bool(with_torch)}, indent=2))
    say("")
    say("Install finished. From now on this starts up offline.")


def missing_packages(with_torch):
    """Import names of the packages that are not importable here."""
    names = [("numpy", "numpy"), ("pandas", "pandas"), ("matplotlib", "matplotlib"),
             ("scipy", "scipy"), ("sklearn", "scikit-learn"), ("skimage", "scikit-image"),
             ("cv2", "opencv-python-headless"), ("jupyterlab", "jupyterlab")]
    if with_torch:
        names.append(("torch", "torch"))
    import importlib.util
    out = []
    for mod, pkg in names:
        if importlib.util.find_spec(mod) is None:
            out.append(pkg)
    return out


# ---------------------------------------------------------------- scheduling

def load_problems():
    path = ROOT / "problems.json"
    if not path.exists():
        die("problems.json is missing, so I cannot tell which problem is today's.",
            "Run python3 tools/build.py to regenerate it.")
    return json.loads(path.read_text())


def start_date():
    """The START_DATE from index.html, so the site and this agree."""
    text = (ROOT / "index.html").read_text(encoding="utf-8")
    m = re.search(r'START_DATE\s*=\s*"(\d{4})-(\d{2})-(\d{2})"', text)
    if not m:
        die("could not find START_DATE in index.html.")
    return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))


def schedule(count):
    """One problem per weekday from START_DATE, exactly as the site does it."""
    out = []
    day = start_date()
    guard = 0
    while len(out) < count and guard < 4000:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
        guard += 1
    return out


def due_index(dates):
    today = date.today()
    idx = -1
    for i, d in enumerate(dates):
        if d <= today:
            idx = i
    return idx


def print_schedule(problems, dates):
    idx = due_index(dates)
    say("")
    say("  #  date          id     problem")
    for i, p in enumerate(problems):
        mark = " <- today" if i == idx else ""
        say("%3d  %s  %-5s  %s%s"
            % (i + 1, dates[i].strftime("%a %d %b %Y"), p["id"], p["title"], mark))
    say("")
    if idx < 0:
        say("The schedule starts on %s. Until then, use a number: python start.py 1"
            % dates[0].strftime("%d %b %Y"))
    else:
        say("Today's problem is number %d. Open it with: python start.py" % (idx + 1))


# ------------------------------------------------------------------ launching

def jupyter_candidates():
    scripts = VENV / ("Scripts" if platform.system() == "Windows" else "bin")
    exe = ".exe" if platform.system() == "Windows" else ""
    out = []
    for name in ("jupyter-lab", "jupyter"):
        cand = scripts / (name + exe)
        if cand.exists():
            out.append([str(cand)] + ([] if name == "jupyter-lab" else ["lab"]))
    out.append([sys.executable, "-m", "jupyter", "lab"])
    out.append([sys.executable, "-m", "jupyterlab"])
    return out


def launch_jupyter(notebook):
    step("Starting JupyterLab")
    say("   A browser tab should open with the notebook in it.")
    say("   Leave this window open while you work. Press Ctrl+C twice here to stop.")
    say("")
    last = 127
    for base in jupyter_candidates():
        last = run(base + [str(notebook)])
        if last != 127:
            return last
    die("could not start JupyterLab.",
        "Try reinstalling with: python start.py --setup")


# ---------------------------------------------------------------------- main

def main(argv):
    if sys.version_info < MIN_PYTHON:
        die("this needs Python %d.%d or newer, and you are running %s."
            % (MIN_PYTHON[0], MIN_PYTHON[1], platform.python_version()),
            "Install a current Python from python.org and try again.")

    in_venv_flag = "--in-venv" in argv
    argv = [a for a in argv if a != "--in-venv"]
    use_venv = "--no-venv" not in argv
    argv = [a for a in argv if a != "--no-venv"]
    with_torch = "--no-torch" not in argv
    argv = [a for a in argv if a != "--no-torch"]
    force = "--reinstall" in argv
    argv = [a for a in argv if a != "--reinstall"]

    want_list = "--list" in argv
    want_browse = "--browse" in argv
    want_setup = "--setup" in argv
    want_found = "--foundations" in argv
    dry_run = "--dry-run" in argv
    argv = [a for a in argv
            if a not in ("--list", "--browse", "--setup", "--dry-run",
                         "--foundations")]

    number = None
    for a in argv:
        if a.startswith("-"):
            die("I do not understand the option %r." % a,
                "Run python start.py --help to see the options.")
        if not a.isdigit():
            die("I expected a problem number, and got %r." % a)
        number = int(a)

    problems = load_problems()
    dates = schedule(len(problems))

    if want_list:
        print_schedule(problems, dates)
        return 0

    if want_browse:
        url = (ROOT / "browse.html").as_uri()
        say("Opening the browse page: " + url)
        if not dry_run:
            webbrowser.open(url)
        return 0

    # Work out which notebook before touching any packages, so a mistake here
    # does not cost you a download.
    if want_found:
        notebook = ROOT / "reference" / "foundations.ipynb"
        if not notebook.exists():
            die("reference/foundations.ipynb is missing.",
                "Run python3 tools/build_foundations.py to regenerate it.")
        say("The foundations notebook: Python, numpy, pandas and matplotlib "
            "from the ground up.")
        say("  " + str(notebook))
        if use_venv and not in_venv_flag and not in_our_venv():
            if dry_run:
                say("(dry run) would create %s and re-run inside it" % VENV)
                return 0
            make_venv()
            reexec_in_venv(sys.argv[1:])
        if dry_run:
            say("(dry run) would launch JupyterLab on the notebook above")
            return 0
        install_packages(with_torch, force=force)
        return launch_jupyter(notebook)

    if number is None:
        idx = due_index(dates)
        if idx < 0:
            say("The schedule starts on %s, which is in the future."
                % dates[0].strftime("%d %b %Y"))
            say("Opening problem 1 instead. Use python start.py --list to see the rest.")
            idx = 0
    else:
        if not 1 <= number <= len(problems):
            die("there are %d problems, so %d is not one of them." % (len(problems), number))
        idx = number - 1

    problem = problems[idx]
    notebook = ROOT / problem["path"]
    if not notebook.exists():
        die("the notebook %s is missing." % problem["path"],
            "Run python3 tools/build.py to regenerate the notebooks.")

    say("Problem %d of %d, %s" % (idx + 1, len(problems), problem["id"]))
    say("  " + problem["title"])
    say("  " + problem["topic"])
    say("  about " + problem["est"])
    say("  " + str(notebook))

    if use_venv and not in_venv_flag and not in_our_venv():
        if dry_run:
            say("")
            say("(dry run) would create %s and re-run inside it" % VENV)
        else:
            make_venv()
            reexec_in_venv(sys.argv[1:])

    if dry_run:
        say("")
        say("(dry run) would install: %s"
            % (", ".join(missing_packages(with_torch)) or "nothing, all present"))
        say("(dry run) would launch JupyterLab on the notebook above")
        return 0

    install_packages(with_torch, force=force)

    if want_setup:
        say("")
        say("Setup done. Run python start.py to open today's problem.")
        return 0

    still_missing = missing_packages(with_torch)
    if still_missing:
        die("these packages are still missing after the install: %s"
            % ", ".join(still_missing),
            "Try: python start.py --reinstall")

    return launch_jupyter(notebook)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--help" in args or "-h" in args:
        print(__doc__)
        sys.exit(0)
    sys.exit(main(args))
