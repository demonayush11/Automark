@echo off
cd /d %~dp0
if not exist .venv (
  echo First run: creating environment...
  python -m venv .venv
)
echo Checking packages (first time downloads about 1 GB)...
.venv\Scripts\python -m pip install -q -r requirements.txt
.venv\Scripts\python app.py
pause
