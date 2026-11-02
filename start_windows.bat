@echo off
cd /d %~dp0
if not exist .venv (
  echo First run: installing... this takes a few minutes.
  python -m venv .venv
  .venv\Scripts\python -m pip install -r requirements.txt
)
.venv\Scripts\python app.py
pause
