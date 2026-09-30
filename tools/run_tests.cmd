@echo off

python "%~dp0run_checks.py" %*

exit /b %errorlevel%

