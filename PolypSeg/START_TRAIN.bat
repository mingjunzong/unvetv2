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

if not exist "%PY%" (
  echo.
  echo [ERROR] 找不到 python: %PY%
  echo         用记事本打开本文件，改上面那三行路径再运行。
  echo.
  pause
  exit /b 1
)
if not exist "%DATA%\TrainDataset\images" (
  echo.
  echo [ERROR] 找不到训练数据: %DATA%\TrainDataset\images
  echo         先跑一次:  python make_data_layout.py --help
  echo.
  pause
  exit /b 1
)

if not exist "%SAVE%" mkdir "%SAVE%"

echo ==============================================================
echo   U-Net v2 / PolypSeg   -   TRAINING   (RTX 3090)
echo --------------------------------------------------------------
echo   code  : %~dp0
echo   data  : %DATA%
echo   save  : %SAVE%
echo --------------------------------------------------------------
echo   recipe: Adam  lr 1e-3  poly decay 0.9  300 epochs
echo           batch 16  fp32 (no AMP)  TF32 on  workers 8
echo           validate every 5 epochs
echo --------------------------------------------------------------
echo   results : %SAVE%\UNetV2\results_log.txt
echo   best    : %SAVE%\UNetV2\best\best.pth
echo ==============================================================
echo.

"%PY%" -u Train.py --train_path "%DATA%\TrainDataset" --test_path "%DATA%\TestDataset" --train_save "%SAVE%" --epoch 300 --batchsize 16 --lr 1e-3 --optimizer Adam --scheduler poly --poly_power 0.9 --val_every 5 --num_workers 8 --device auto --seed 1234 %*

echo.
echo ==============================================================
echo   Finished.  exit code = %ERRORLEVEL%
echo ==============================================================
pause
