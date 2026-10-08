@echo off
setlocal
cd /d "%~dp0"

rem ============================================================
rem   >>> 换机器只需要改下面这三行 <<<
rem ============================================================
set "PY=C:\Users\ming\anaconda3\envs\unetv2\python.exe"
set "DATA=D:\unetv2\unetv2_data"
set "SAVE=D:\unetv2\runs"
rem ============================================================

set "CKPT=%SAVE%\UNetV2\best\best.pth"

if not exist "%PY%" (
  echo.
  echo [ERROR] 找不到 python: %PY%
  echo         用记事本打开本文件，改上面那三行路径再运行。
  echo.
  pause
  exit /b 1
)

echo ==============================================================
echo   U-Net v2 / PolypSeg   -   TEST
echo --------------------------------------------------------------
echo   checkpoint : %CKPT%
echo ==============================================================
echo.

if not exist "%CKPT%" (
  echo [ERROR] 找不到权重文件，请先跑 START_TRAIN.bat。
  echo.
  pause
  exit /b 1
)

"%PY%" -u Test.py --test_path "%DATA%\TestDataset" --pth_path "%CKPT%" --save_path "%SAVE%\test_result" --save_masks %*

echo.
echo ==============================================================
echo   results : %SAVE%\test_result\test_results.txt
echo   masks   : %SAVE%\test_result\CVC-ClinicDB\
echo ==============================================================
pause
