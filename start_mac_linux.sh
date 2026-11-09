#!/bin/sh
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
echo "Checking packages (first time downloads about 1 GB)..."
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python app.py
