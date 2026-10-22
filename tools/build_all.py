"""Rebuild every generated file, in the right order.

    python3 tools/build_all.py

Run this after changing anything in tools/specs/ or tools/function_catalog.py.
Then run tools/verify.py, tools/negative_test.py and tools/check_dashes.py.
"""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

STEPS = [
    ("problem and solution notebooks", "tools/build.py"),
    ("foundations notebook", "tools/build_foundations.py"),
    ("function reference docx", "tools/build_glossary.py"),
    ("browse page", "tools/build_browse.py"),
]


def main():
    for label, script in STEPS:
        print("=== %s ===" % label)
        result = subprocess.run([sys.executable, str(ROOT / script)], cwd=str(ROOT))
        if result.returncode != 0:
            print("")
            print("FAILED at: %s" % script)
            return result.returncode
        print("")
    print("Everything rebuilt. Now run:")
    print("  python3 tools/verify.py")
    print("  python3 tools/negative_test.py")
    print("  python3 tools/check_dashes.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
