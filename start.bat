@echo off
title Seedance2API - ByteDance Dreamina AI Proxy (:8090)
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" -m uvicorn server:app --host 127.0.0.1 --port 8090 --reload
pause
