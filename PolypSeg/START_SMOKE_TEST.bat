@echo off
setlocal
cd /d "%~dp0"

rem ============================================================
rem   >>> 换机器只需要改下面这一行 <<<
rem ============================================================
set "PY=C:\Users\ming\anaconda3\envs\unetv2\python.exe"
rem ============================================================

if not exist "%PY%" (
  echo.
  echo [ERROR] 找不到 python: %PY%
  echo         用记事本打开本文件，改上面那一行路径再运行。
  echo.
  pause
  exit /b 1
)

echo ==============================================================
echo   U-Net v2 / PolypSeg   -   SELF CHECK  (no dataset needed)
echo ==============================================================
echo.

"%PY%" -u smoke_test.py %*

echo.
pause
