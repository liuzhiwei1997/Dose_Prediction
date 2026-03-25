@echo off
setlocal enabledelayedexpansion

REM Windows CMD end-to-end DVH optimization helper.
REM Usage:
REM   scripts\run_dvh_optimization_windows.bat

cd /d %~dp0\..

if "%DOSE_PREDICTION_DATA_ROOT%"=="" set DOSE_PREDICTION_DATA_ROOT=%cd%
if "%DOSE_PREDICTION_OUTPUT_ROOT%"=="" set DOSE_PREDICTION_OUTPUT_ROOT=%cd%\runs
if "%DOSE_PREDICTION_NUM_WORKERS%"=="" set DOSE_PREDICTION_NUM_WORKERS=4
if "%DOSE_PREDICTION_BATCH_SIZE%"=="" set DOSE_PREDICTION_BATCH_SIZE=1
if "%DOSE_PREDICTION_SW_BATCH_SIZE%"=="" set DOSE_PREDICTION_SW_BATCH_SIZE=1
if "%DOSE_PREDICTION_LOGGER%"=="" set DOSE_PREDICTION_LOGGER=csv
if "%DOSE_PREDICTION_USE_BNB%"=="" set DOSE_PREDICTION_USE_BNB=0

echo [1/3] Stage-1 coarse DVH tuning...
python scripts/tune_dvh_hparams.py ^
  --max-epochs 20 ^
  --hotspot-weights "0.5,0.75,1.0" ^
  --coldspot-weights "0.2,0.35,0.5" ^
  --hotspot-quantiles "0.98,0.99" ^
  --coldspot-quantiles "0.05,0.10" ^
  --dose-weight 0.5 ^
  --output-root runs/DosePrediction/dvh_tuning_stage1
if errorlevel 1 exit /b 1

echo [2/3] Stage-2 fine DVH tuning...
python scripts/tune_dvh_hparams.py ^
  --max-epochs 35 ^
  --hotspot-weights "0.85,1.0,1.15" ^
  --coldspot-weights "0.25,0.35,0.45" ^
  --hotspot-quantiles "0.985,0.99" ^
  --coldspot-quantiles "0.08,0.10,0.12" ^
  --dose-weight 0.5 ^
  --output-root runs/DosePrediction/dvh_tuning_stage2
if errorlevel 1 exit /b 1

echo [3/3] Print best config...
python -c "import json, pathlib; p=pathlib.Path('runs/DosePrediction/dvh_tuning_stage2/dvh_tuning_results.json'); assert p.exists(), f'Missing tuning summary: {p}'; data=json.loads(p.read_text(encoding='utf-8')); print(json.dumps(data['best'], indent=2, ensure_ascii=False))"
if errorlevel 1 exit /b 1

echo Done. You can now use the best config for full training.
endlocal
