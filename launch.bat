@echo off
rem Quillcut Studio: double-click to set up (first run only) and open the app.
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\quillcut.exe" goto run

python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if errorlevel 1 (
  echo Python 3.11 or newer is required. Install it from https://www.python.org/downloads/ and try again.
  goto fail
)
where ffmpeg >nul 2>&1
if errorlevel 1 (
  echo ffmpeg was not found. Install it from https://ffmpeg.org/download.html and make sure it is on your PATH.
  goto fail
)

echo First run: setting up Quillcut. This downloads a few GB and can take several minutes.
python -m venv .venv
if errorlevel 1 goto fail

nvidia-smi >nul 2>&1
if errorlevel 1 (
  echo No NVIDIA GPU found, installing the CPU build of PyTorch. Reels will render slower.
  set TORCH_INDEX=https://download.pytorch.org/whl/cpu
) else (
  set TORCH_INDEX=https://download.pytorch.org/whl/cu126
)

".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install torch==2.11.0 torchvision==0.26.0 --index-url %TORCH_INDEX%
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install -e .
if errorlevel 1 goto fail

:run
echo Starting Quillcut. Keep this window open while you use the app.
".venv\Scripts\quillcut.exe" web
goto end

:fail
echo.
echo Setup did not finish. Fix the problem above and run start.bat again.
pause
exit /b 1

:end
endlocal
