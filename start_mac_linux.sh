#!/bin/sh
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "First run: installing... this takes a few minutes."
  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
fi
.venv/bin/python app.py
