@echo off
REM brainrotbot -- single entry point. Runs the full pipeline in the conda env.
REM Double-click this file, or run it from a terminal. Extra args pass through, e.g.:
REM   run.bat --top-k 3
echo Starting brainrotbot...
REM Runs in the brainrotbot312 (Python 3.12) env: Kokoro TTS needs Python <3.13.
call "C:\ProgramData\anaconda3\condabin\conda.bat" run -n brainrotbot312 --no-capture-output python -m brainrotbot.pipeline %*
echo.
echo brainrotbot finished (exit code %errorlevel%).
pause
