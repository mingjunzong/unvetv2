@echo off
setlocal
cd /d "%~dp0"

rem ============================================================
rem   >>> 换机器只需要改下面这三行 <<<
rem ============================================================
set "PY=C:\Users\ming\anaconda3\envs\unetv2\python.exe"
set "DATA=D:\unetv2\unetv2_data"
set "SAVE=D:\unetv2\verify"
rem ============================================================

if not exist "%PY%" (
  echo.
  echo [ERROR] 找不到 python: %PY%
  echo         用记事本打开本文件，改上面那三行路径再运行。
  echo.
  pause
  exit /b 1
)

if not exist "%SAVE%" mkdir "%SAVE%"

echo ==============================================================
echo   U-Net v2 / PolypSeg   -   QUICK VERIFY  (only 5 epochs)
echo --------------------------------------------------------------
echo   先跑这个！确认 loss 在降、dsc 在涨，再去跑 300 epoch。
echo   如果 5 轮后 dsc 还是 0.00，那百分之百是数据问题，
echo   跑 300 轮也是白跑，先去看 check_data.py 的结果。
echo ==============================================================
echo.

"%PY%" -u Train.py --train_path "%DATA%\TrainDataset" --test_path "%DATA%\TestDataset" --train_save "%SAVE%" --epoch 5 --batchsize 16 --lr 1e-3 --optimizer Adam --scheduler poly --poly_power 0.9 --val_every 1 --num_workers 8 --device auto --seed 1234 %*

echo.
echo ==============================================================
echo   Finished.  exit code = %ERRORLEVEL%
echo   看 dsc 有没有离开 0.00 : %SAVE%\UNetV2\results_log.txt
echo ==============================================================
pause
