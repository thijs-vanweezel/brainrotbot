@echo off
REM brainrotbot -- single entry point. Runs the full pipeline in the conda env.
REM Double-click this file, or run it from a terminal. Extra args pass through, e.g.:
REM   run.bat --top-k 3
echo Starting brainrotbot...
call "C:\ProgramData\anaconda3\condabin\conda.bat" run -n brainrotbot --no-capture-output python -m brainrotbot.pipeline %*
echo.
echo brainrotbot finished (exit code %errorlevel%).
pause
