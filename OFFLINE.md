# Working offline, with no Colab

Colab is convenient and borrowed. Everything here also runs on your own
computer, with one command, and once it is set up it needs no internet at all.

Nothing in the notebooks changes. The same problem files, the same `check()`,
the same solutions. Only where they run is different, so you can switch back
and forth whenever you like.

## Once, to set up

You need Python 3.9 or newer. To find out whether you have it, open a terminal
(on Windows, Command Prompt or PowerShell) and type:

```
python --version
```

If that says `Python 3.9` or higher, you are ready. If it says the command is
not found, or shows a 2.x version, install Python from
[python.org/downloads](https://www.python.org/downloads/). On Windows, tick
**Add python.exe to PATH** on the first screen of the installer, which is easy
to miss and causes most of the trouble people have afterwards.

If you have Anaconda, you already have Python and can use it instead. See the
Anaconda section at the bottom.

### Get the folder

Either download the repository as a zip from GitHub and unzip it somewhere you
can find again, or, if you have git:

```
git clone https://github.com/dc00086-glitch/internship-prep.git
cd internship-prep
```

### Run it

In a terminal, in that folder:

```
python start.py
```

On Windows you can instead double click **start_windows.bat**. On a Mac,
double click **start_mac_linux.command**.

The first run needs internet. It makes a folder called `.venv` inside the
project, installs numpy, pandas, matplotlib, scipy, scikit-learn,
scikit-image, OpenCV, PyTorch and JupyterLab into it, and then opens today's
problem in your browser. Expect a few minutes, a few hundred megabytes of
download, and a bit over a gigabyte on disk once it is unpacked.

One caveat on Linux. PyTorch is fetched from its CPU only download site, and
if a proxy blocks that site the launcher falls back to the ordinary package
index, which on Linux ships a build carrying about 5 GB of GPU libraries that
nothing here uses. It works, it is just large. `python start.py --no-torch`
avoids it entirely until you reach week 6.

Every run after that is offline and takes a couple of seconds.

Nothing is installed into your system Python and nothing outside this folder
is touched. Deleting the `.venv` folder undoes all of it.

## Every day after that

```
python start.py
```

That picks today's problem from the same schedule the website uses, one per
weekday from the start date in `index.html`, and opens it in JupyterLab.

Other things you can ask for:

| Command | What it does |
|---|---|
| `python start.py` | today's problem |
| `python start.py 12` | problem 12, whatever the date |
| `python start.py --list` | the whole schedule with dates, and which one is today's |
| `python start.py --browse` | the browse page, every problem on one page, no Jupyter needed |
| `python start.py --foundations` | the foundations notebook, for a refresher |
| `python start.py --setup` | install everything and stop, for setting up before a flight |
| `python start.py --no-torch` | skip the PyTorch download, which only week 6 needs |
| `python start.py --reinstall` | install the packages again, if something is broken |
| `python start.py --no-venv` | use your current Python instead of making a .venv |

## Working in JupyterLab instead of Colab

Nearly everything is the same. The differences worth knowing:

- **Run a cell** with Shift and Enter, as in Colab.
- **The file browser** is the left hand panel, so you can open the solutions
  folder and the foundations notebook from there.
- **Your work saves to the file on disk**, immediately and with no copy in
  Drive. That is better in every way except one: if you want the original
  blank notebook back, you have to get it from GitHub again, or run
  `python3 tools/build.py`, which rewrites every problem notebook from its
  spec. Both of those discard whatever you had typed, so copy your answers out
  first.
- **Stop it** by closing the browser tab and pressing Ctrl+C twice in the
  terminal window.
- **Plots** appear under the cell exactly as they do in Colab.

## The browse page

`browse.html` is a single page holding every problem's scenario, explanation,
task, setup code, toolkit and parts, with a sidebar to move between days and a
toolkit drawer that stays open while you read. It is plain HTML with
everything embedded, so it works offline by just opening the file, with no
Python running at all.

```
python start.py --browse
```

or open `browse.html` by double clicking it.

Use it to read ahead, to look something up, or to work on paper. The blanks
are not fillable there, which is the point of it.

## If you use Anaconda

You do not need the `.venv` at all. In the Anaconda Prompt, in the project
folder:

```
conda install -y numpy pandas matplotlib scipy scikit-learn scikit-image jupyterlab
pip install opencv-python-headless
conda install -y pytorch cpuonly -c pytorch
python start.py --no-venv
```

`--no-venv` tells the launcher to use the Python it was started with, which is
your conda environment.

## When something goes wrong

**`python: command not found`** on Windows. Python is installed but not on
the PATH. Try `py start.py` instead, which uses the Windows Python launcher.

**`No module named venv`** on Linux. Your distribution splits it out. Either
`sudo apt install python3-venv`, or run `python start.py --no-venv` and
install the packages yourself with
`python3 -m pip install -r requirements.txt`.

**The install fails with a network error.** The first run is the only one that
needs internet. Check the connection and run it again; the install resumes
rather than starting over.

**PyTorch takes forever.** It is the largest download by far, around 200 MB
as the CPU build, and only week 6 uses it. `python start.py --no-torch` skips
it entirely, and you can add it later with `python start.py --setup`. If the
install of everything else worked and only PyTorch failed, the launcher says
so and carries on: the other 35 problems do not need it.

**A notebook says `No module named cv2`** or similar. Something went wrong in
the install. Run `python start.py --reinstall`.

**I want to start a problem over.** Running `python3 tools/build.py` rewrites
every problem notebook from its spec, which gives you the blank versions back
and discards any answers you had typed. Copy them out first.

**My answers and the solution disagree but `check()` says PASS.** Then your
answer is right. The checks test the numbers, not the route, and there is
usually more than one way to get there.

## What this does not need

No account, no login, no Google, no internet after the first run, and no
administrator rights. The data in every problem is generated inside the
notebook from a fixed seed, so there are no files to download and your numbers
always match the ones in the text.
