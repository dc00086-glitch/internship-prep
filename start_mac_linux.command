#!/bin/sh
# Double click this file on a Mac, or run ./start_mac_linux.command in a
# terminal on Linux, to open today's problem. It calls start.py, which
# installs what it needs the first time.

cd "$(dirname "$0")" || exit 1

if command -v python3 >/dev/null 2>&1; then
  python3 start.py "$@"
elif command -v python >/dev/null 2>&1; then
  python start.py "$@"
else
  echo ""
  echo "Python 3 was not found on this computer."
  echo "On a Mac, install it from https://www.python.org/downloads/"
  echo "On Linux, try: sudo apt install python3 python3-venv"
  echo ""
  exit 1
fi
