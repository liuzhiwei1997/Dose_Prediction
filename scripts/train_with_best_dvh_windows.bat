@echo off
setlocal

REM Windows CMD wrapper: inject best DVH tuning params into full training.
REM Usage:
REM   scripts\train_with_best_dvh_windows.bat

cd /d %~dp0\..

if "%DOSE_PREDICTION_NUM_WORKERS%"=="" set DOSE_PREDICTION_NUM_WORKERS=0
if "%DOSE_PREDICTION_USE_CACHE%"=="" set DOSE_PREDICTION_USE_CACHE=0
if "%DOSE_PREDICTION_CACHE_RATE%"=="" set DOSE_PREDICTION_CACHE_RATE=0.0
if "%DOSE_PREDICTION_TRAIN_DIR%"=="" set DOSE_PREDICTION_TRAIN_DIR=provided-data/nifti-train-pats/pt_*
if "%DOSE_PREDICTION_VAL_DIR%"=="" set DOSE_PREDICTION_VAL_DIR=provided-data/nifti-val-pats/pt_*
if "%DOSE_PREDICTION_TEST_DIR%"=="" set DOSE_PREDICTION_TEST_DIR=provided-data/nifti-test-pats/pt_*

if not exist runs\DosePrediction\dvh_tuning_stage2\dvh_tuning_results.json (
  echo [INFO] Missing runs\DosePrediction\dvh_tuning_stage2\dvh_tuning_results.json
  echo [INFO] Running DVH tuning first...
  call scripts\run_dvh_optimization_windows.bat
  if errorlevel 1 exit /b 1
)

python scripts/train_with_best_dvh.py ^
  --best-json runs/DosePrediction/dvh_tuning_stage2/dvh_tuning_results.json ^
  --freeze-epochs 300 ^
  --finetune-epochs 120 ^
  --resume ^
  --output-dir runs/DosePrediction/final_from_best_dvh

endlocal
