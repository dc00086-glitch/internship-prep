"""Build browse.html, one page showing every week and day in full.

    python3 tools/build_browse.py

Everything is embedded, so it needs no network and works offline. New weeks
are picked up automatically from tools/specs.
"""

import html
import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPEC_DIR = ROOT / "tools" / "specs"
OUT = ROOT / "browse.html"

OWNER, REPO, BRANCH = "dc00086-glitch", "internship-prep", "main"

WEEK_THEME = {
    1: "numpy, pandas and matplotlib",
    2: "pandas in depth",
    3: "scipy",
    4: "scikit-image and OpenCV",
    5: "scikit-learn",
    6: "PyTorch basics",
    7: "PK and PD applications",
    8: "PK and PD applications",
}


def md_to_html(text):
    import markdown
    return markdown.markdown(text.strip(), extensions=["tables", "fenced_code", "sane_lists"])


def code_block(text):
    return '<pre class="code"><code>%s</code></pre>' % html.escape(text.strip())


def colab_url(path):
    return "https://colab.research.google.com/github/%s/%s/blob/%s/%s" % (OWNER, REPO, BRANCH, path)


def github_url(path):
    return "https://github.com/%s/%s/blob/%s/%s" % (OWNER, REPO, BRANCH, path)


def load_specs():
    mods = []
    for path in sorted(SPEC_DIR.glob("w*.py")):
        obj = importlib.util.spec_from_file_location("spec_" + path.stem, path)
        mod = importlib.util.module_from_spec(obj)
        obj.loader.exec_module(mod)
        mods.append(mod)
    mods.sort(key=lambda m: (m.WEEK, m.DAY))
    return mods


def build_problem(spec):
    week_dir = "week%02d" % spec.WEEK
    prob_path = "problems/%s/%s.ipynb" % (week_dir, spec.SLUG)
    sol_path = "solutions/%s/%s_solution.ipynb" % (week_dir, spec.SLUG)

    sections = []
    sections.append(('scenario', 'Scenario', md_to_html(spec.SCENARIO_MD)))
    if getattr(spec, "CONCEPT_MD", None):
        title = [l for l in spec.CONCEPT_MD.strip().splitlines() if l.startswith("## ")]
        label = title[0][3:].strip() if title else "How it works"
        sections.append(('concept', label, md_to_html(spec.CONCEPT_MD)))
    sections.append(('task', 'Your task', md_to_html(spec.TASK_MD)))
    sections.append(('setup', 'Setup', code_block(spec.DATA_CODE)))

    toolkit_html = ""
    if getattr(spec, "TOOLKIT_MD", None):
        toolkit_html = md_to_html(spec.TOOLKIT_MD)
    toolkit_full = toolkit_html
    if getattr(spec, "TOOLKIT_CODE", None):
        toolkit_full += code_block(spec.TOOLKIT_CODE)
    if toolkit_full:
        sections.append(('toolkit', 'Toolkit', toolkit_full))

    for part in spec.PARTS:
        body = md_to_html(part["prompt_md"]) + code_block(part["starter"])
        sections.append(('part' + part["label"], 'Part ' + part["label"], body))

    return {
        "id": spec.PID, "week": spec.WEEK, "day": spec.DAY,
        "title": spec.TITLE, "topic": spec.TOPIC, "est": spec.EST,
        "colab": colab_url(prob_path), "github": github_url(prob_path),
        "solution": github_url(sol_path),
        "toolkit": toolkit_html,
        "sections": [{"key": k, "label": l, "html": h} for k, l, h in sections],
    }


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Browse all problems</title>
<meta name="description" content="Every week and day of the internship-prep practice set, in one page.">
<style>
  :root {
    --bg: #f7f8fa; --card: #ffffff; --ink: #16181d; --ink-soft: #5b6472;
    --line: #e3e6eb; --accent: #1b5e9c; --accent-ink: #ffffff; --badge: #eef3f9;
    --code-bg: #f4f6f8; --code-ink: #1f2937; --warn: #9a3412;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #14161a; --card: #1c1f25; --ink: #e8eaee; --ink-soft: #9aa4b2;
      --line: #2c3138; --accent: #5ea2e0; --accent-ink: #10131a; --badge: #252a32;
      --code-bg: #15181d; --code-ink: #d6dae0; --warn: #f0a070;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .layout { display: flex; min-height: 100vh; }

  /* sidebar */
  aside {
    width: 272px; flex: 0 0 272px; border-right: 1px solid var(--line);
    background: var(--card); padding: 20px 0 48px; position: sticky; top: 0;
    height: 100vh; overflow-y: auto;
  }
  aside h1 { font-size: 1rem; margin: 0 18px 4px; }
  aside .sub { font-size: 0.8rem; color: var(--ink-soft); margin: 0 18px 16px; }
  aside a.home { display: block; margin: 0 18px 16px; font-size: 0.82rem; color: var(--accent); text-decoration: none; }
  aside a.home:hover { text-decoration: underline; }
  .wk { margin-top: 10px; }
  .wk-h {
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em;
    font-weight: 700; color: var(--ink-soft); padding: 8px 18px 4px;
  }
  .wk-h span { text-transform: none; letter-spacing: 0; font-weight: 400; }
  .d {
    display: block; padding: 7px 18px; font-size: 0.85rem; color: var(--ink);
    text-decoration: none; border-left: 3px solid transparent; cursor: pointer;
  }
  .d:hover { background: var(--badge); }
  .d.on { border-left-color: var(--accent); background: var(--badge); font-weight: 600; }
  .d code { font-size: 0.78rem; color: var(--ink-soft); margin-right: 7px; }
  .d.on code { color: var(--accent); }

  /* main */
  main { flex: 1 1 auto; min-width: 0; padding: 26px 32px 120px; max-width: 900px; }
  .pid {
    display: inline-block; font: 600 0.74rem/1 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    background: var(--badge); color: var(--ink-soft); border: 1px solid var(--line);
    border-radius: 5px; padding: 5px 7px;
  }
  h2.title { font-size: 1.4rem; margin: 10px 0 6px; line-height: 1.3; }
  .meta { color: var(--ink-soft); font-size: 0.88rem; margin: 0 0 16px; }
  .btns { display: flex; flex-wrap: wrap; gap: 9px; margin-bottom: 8px; }
  a.btn {
    display: inline-block; text-decoration: none; font-weight: 600; font-size: 0.88rem;
    padding: 9px 15px; border-radius: 8px; border: 1px solid var(--accent);
    background: var(--accent); color: var(--accent-ink);
  }
  a.btn.ghost { background: transparent; color: var(--accent); }
  a.btn.quiet { background: transparent; color: var(--ink-soft); border-color: var(--line); }
  a.btn:hover { filter: brightness(1.08); }

  .jump {
    position: sticky; top: 0; z-index: 5; background: var(--bg);
    padding: 12px 0 10px; border-bottom: 1px solid var(--line); margin-bottom: 8px;
    display: flex; flex-wrap: wrap; gap: 7px;
  }
  .jump a {
    font-size: 0.78rem; text-decoration: none; color: var(--ink-soft);
    border: 1px solid var(--line); border-radius: 999px; padding: 5px 11px; background: var(--card);
  }
  .jump a:hover { color: var(--ink); border-color: var(--accent); }

  section.blk { border-bottom: 1px solid var(--line); padding: 20px 0 6px; scroll-margin-top: 62px; }
  section.blk > h3 {
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.09em;
    color: var(--ink-soft); margin: 0 0 10px; font-weight: 700;
  }
  .body h2 { font-size: 1.12rem; margin: 18px 0 8px; }
  .body h3 { font-size: 1rem; margin: 16px 0 6px; }
  .body p { margin: 0 0 11px; }
  .body ul, .body ol { margin: 0 0 11px; padding-left: 22px; }
  .body li { margin-bottom: 4px; }
  .body blockquote {
    margin: 0 0 12px; padding: 8px 14px; border-left: 3px solid var(--accent);
    background: var(--badge); color: var(--ink-soft);
  }
  .body code {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.86em; background: var(--badge); padding: 1px 5px; border-radius: 4px;
  }
  .body table { border-collapse: collapse; width: 100%; margin: 0 0 14px; font-size: 0.88rem; }
  .body th, .body td { border: 1px solid var(--line); padding: 7px 9px; text-align: left; vertical-align: top; }
  .body th { background: var(--badge); font-weight: 600; }
  .body td code { background: transparent; padding: 0; }
  pre.code {
    background: var(--code-bg); color: var(--code-ink); border: 1px solid var(--line);
    border-radius: 8px; padding: 13px 15px; overflow-x: auto; margin: 0 0 14px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.8rem; line-height: 1.5;
  }
  pre.code code { background: none; padding: 0; font-size: inherit; }

  .pager { display: flex; gap: 10px; margin-top: 26px; }
  .pager a, .pager span {
    font-size: 0.85rem; text-decoration: none; color: var(--ink-soft);
    border: 1px solid var(--line); border-radius: 8px; padding: 9px 13px; background: var(--card);
    cursor: pointer;
  }
  .pager span { opacity: 0.4; cursor: default; }
  .pager a:hover { color: var(--ink); border-color: var(--accent); }

  /* toolkit drawer */
  #tk-btn {
    position: fixed; right: 20px; bottom: 20px; z-index: 20;
    background: var(--accent); color: var(--accent-ink); border: none;
    border-radius: 999px; padding: 13px 20px; font-size: 0.88rem; font-weight: 600;
    cursor: pointer; box-shadow: 0 3px 14px rgba(0,0,0,0.22); font-family: inherit;
  }
  #tk-btn:hover { filter: brightness(1.08); }
  #tk {
    position: fixed; top: 0; right: 0; width: min(540px, 94vw); height: 100vh; z-index: 30;
    background: var(--card); border-left: 1px solid var(--line); overflow-y: auto;
    padding: 20px 24px 60px; transform: translateX(102%); transition: transform 0.18s ease;
    box-shadow: -4px 0 24px rgba(0,0,0,0.14);
  }
  #tk.open { transform: translateX(0); }
  #tk .tk-top { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
  #tk h3 { margin: 0; font-size: 0.95rem; }
  #tk button {
    background: transparent; border: 1px solid var(--line); border-radius: 7px;
    color: var(--ink-soft); padding: 5px 11px; cursor: pointer; font-family: inherit; font-size: 0.82rem;
  }
  #tk button:hover { color: var(--ink); }

  #menu-btn { display: none; }
  @media (max-width: 860px) {
    .layout { flex-direction: column; }
    aside { position: static; width: auto; height: auto; flex: none; border-right: none;
            border-bottom: 1px solid var(--line); padding-bottom: 14px; }
    main { padding: 18px 16px 120px; }
  }
</style>
</head>
<body>
<div class="layout">
  <aside>
    <h1>Browse all problems</h1>
    <p class="sub" id="aside-sub"></p>
    <a class="home" href="./">Back to today's problem</a>
    <nav id="nav"></nav>
  </aside>
  <main id="main"></main>
</div>

<button id="tk-btn" type="button">Toolkit</button>
<div id="tk">
  <div class="tk-top"><h3 id="tk-title">Toolkit</h3><button type="button" id="tk-close">Close</button></div>
  <div class="body" id="tk-body"></div>
</div>

<script>
  // === PROBLEMS START ===
  const PROBLEMS = __PROBLEMS__;
  // === PROBLEMS END ===
  const THEMES = __THEMES__;

  const nav = document.getElementById("nav");
  const main = document.getElementById("main");
  const tk = document.getElementById("tk");
  const tkBody = document.getElementById("tk-body");
  const tkTitle = document.getElementById("tk-title");

  const N_WEEKS = new Set(PROBLEMS.map(function (p) { return p.week; })).size;
  document.getElementById("aside-sub").textContent =
    PROBLEMS.length + " problems across " + N_WEEKS + (N_WEEKS === 1 ? " week" : " weeks");

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function buildNav() {
    const weeks = {};
    PROBLEMS.forEach(function (p) { (weeks[p.week] = weeks[p.week] || []).push(p); });
    let html = "";
    Object.keys(weeks).sort(function (a, b) { return a - b; }).forEach(function (wk) {
      const theme = THEMES[wk] ? "<span> &middot; " + esc(THEMES[wk]) + "</span>" : "";
      html += '<div class="wk"><div class="wk-h">Week ' + wk + theme + "</div>";
      weeks[wk].forEach(function (p) {
        html += '<a class="d" data-id="' + p.id + '" href="#' + p.id + '">' +
                "<code>" + esc(p.id) + "</code>" + esc(p.title) + "</a>";
      });
      html += "</div>";
    });
    nav.innerHTML = html;
  }

  function indexOf(id) {
    for (let i = 0; i < PROBLEMS.length; i++) if (PROBLEMS[i].id === id) return i;
    return -1;
  }

  function render(id) {
    const i = indexOf(id);
    if (i < 0) return;
    const p = PROBLEMS[i];

    let html = '<span class="pid">' + esc(p.id) + "</span>" +
      '<h2 class="title">' + esc(p.title) + "</h2>" +
      '<p class="meta">' + esc(p.topic) + "<br>Week " + p.week + ", day " + p.day +
      " &middot; " + esc(p.est) + "</p>" +
      '<div class="btns">' +
        '<a class="btn" href="' + p.colab + '">Open in Colab</a>' +
        '<a class="btn ghost" href="' + p.github + '">View on GitHub</a>' +
        '<a class="btn quiet" href="' + p.solution + '">Solution</a>' +
      "</div>";

    html += '<div class="jump">';
    p.sections.forEach(function (s) {
      html += '<a href="#sec-' + s.key + '">' + esc(s.label) + "</a>";
    });
    html += "</div>";

    p.sections.forEach(function (s) {
      html += '<section class="blk" id="sec-' + s.key + '">' +
              "<h3>" + esc(s.label) + '</h3><div class="body">' + s.html + "</div></section>";
    });

    html += '<div class="pager">';
    html += i > 0
      ? '<a href="#' + PROBLEMS[i - 1].id + '">Previous, ' + esc(PROBLEMS[i - 1].id) + "</a>"
      : "<span>Previous</span>";
    html += i < PROBLEMS.length - 1
      ? '<a href="#' + PROBLEMS[i + 1].id + '">Next, ' + esc(PROBLEMS[i + 1].id) + "</a>"
      : "<span>Next</span>";
    html += "</div>";

    main.innerHTML = html;
    main.scrollTop = 0;
    window.scrollTo(0, 0);

    Array.prototype.forEach.call(nav.querySelectorAll(".d"), function (a) {
      a.classList.toggle("on", a.getAttribute("data-id") === id);
    });

    tkTitle.textContent = "Toolkit, " + p.id;
    tkBody.innerHTML = p.toolkit || "<p>This problem has no toolkit section.</p>";
    document.title = p.id + ", " + p.title;
  }

  function fromHash() {
    const id = location.hash.replace(/^#/, "");
    render(indexOf(id) >= 0 ? id : PROBLEMS[0].id);
  }

  document.getElementById("tk-btn").addEventListener("click", function () {
    tk.classList.toggle("open");
  });
  document.getElementById("tk-close").addEventListener("click", function () {
    tk.classList.remove("open");
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") tk.classList.remove("open");
  });
  window.addEventListener("hashchange", fromHash);

  buildNav();
  fromHash();
</script>
</body>
</html>
"""


def main():
    try:
        import markdown  # noqa: F401
    except ImportError:
        raise SystemExit("needs the markdown package: pip install markdown")

    specs = load_specs()
    if not specs:
        raise SystemExit("no specs found")
    problems = [build_problem(s) for s in specs]

    page = PAGE.replace("__PROBLEMS__", json.dumps(problems, indent=1))
    page = page.replace("__THEMES__", json.dumps({str(k): v for k, v in WEEK_THEME.items()}))
    OUT.write_text(page, encoding="utf-8")

    size_kb = round(OUT.stat().st_size / 1024)
    weeks = sorted({s.WEEK for s in specs})
    print("wrote browse.html (%d problems, week %s, %d KB)"
          % (len(problems), ", ".join(str(w) for w in weeks), size_kb))
    return 0


if __name__ == "__main__":
    sys.exit(main())
