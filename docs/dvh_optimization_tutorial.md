# DosePrediction：DVH 指标优化实战教程（继续优化，不必从零重来）

> 结论先说：**建议接着优化，不建议直接从零重训**。  
> 你当前结果已具备可优化基础，先做“自动调参 + 定向微调”，通常能更快把 `D_0.1_cc`、`D99` 拉下来。

---

## 1. 什么时候“继续优化”，什么时候“重练”

### 建议继续优化（你现在属于这类）
- 训练流程能稳定收敛；
- `Dose score` 和 `DVH score` 已在合理区间但还有提升空间；
- 问题集中在少数指标（如 `D_0.1_cc`、`D99`）而非全面崩坏。

### 才考虑重练（从零）
- 数据预处理/对齐有错误（spacing、mask、剂量归一化）；
- 模型训练不稳定（loss 发散、NaN）；
- 架构或数据分布发生大变化（例如改任务、改器官标注体系）。

---

## 2. 一键流程概览

推荐分三步：

1. **固定基线评估**（确认当前模型分数）
2. **验证集 DVH 自动调参**（先找对 `D_0.1_cc`/`D99` 更友好的 loss 权重）
3. **用最优参数做完整训练 + 推理评估**

### 2.1 训练/验证/测试集怎么用（你这次重点）

- `nifti-train-pats`：只用于**参数学习**（反向传播）。
- `nifti-val-pats`：只用于**调参和早停判断**（不参与梯度更新）。
- `nifti-test-pats`：只用于**最终一次报告**（不要在调参阶段反复看它）。

当前代码已支持通过环境变量显式指定三者目录：

```bash
export DOSE_PREDICTION_TRAIN_DIR='provided-data/nifti-train-pats/pt_*'
export DOSE_PREDICTION_VAL_DIR='provided-data/nifti-val-pats/pt_*'
export DOSE_PREDICTION_TEST_DIR='provided-data/nifti-test-pats/pt_*'
```

---

## 3. 环境准备

在仓库根目录执行：

```bash
cd /workspace/Dose_Prediction
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

可选（强制 CPU，调试时用）：

```bash
export DOSE_PREDICTION_FORCE_CPU=1
```

> Windows 用户提示：如果你在 `cmd` 里看到  
> `bash 不是内部或外部命令`，请改用下面的 `python` 命令或 `scripts/run_dvh_optimization_windows.bat`，不要直接执行 `.sh`。
> 若出现 `MemoryError`，请设置：
> `set DOSE_PREDICTION_NUM_WORKERS=0`、`set DOSE_PREDICTION_USE_CACHE=0`、`set DOSE_PREDICTION_CACHE_RATE=0.0`。

---

## 4. 第一步：先拿到可复现基线

```bash
cd /workspace/Dose_Prediction
source .venv/bin/activate

export DOSE_PREDICTION_DATA_ROOT=/workspace/Dose_Prediction
export DOSE_PREDICTION_OUTPUT_ROOT=/workspace/Dose_Prediction/runs
export DOSE_PREDICTION_TRAIN_DIR='provided-data/nifti-train-pats/pt_*'
export DOSE_PREDICTION_VAL_DIR='provided-data/nifti-val-pats/pt_*'
export DOSE_PREDICTION_TEST_DIR='provided-data/nifti-test-pats/pt_*'
export DOSE_PREDICTION_NUM_WORKERS=4
export DOSE_PREDICTION_BATCH_SIZE=1
export DOSE_PREDICTION_SW_BATCH_SIZE=1
export DOSE_PREDICTION_LOGGER=csv
export DOSE_PREDICTION_USE_BNB=0
```

如果你已有预测结果目录（比如 `runs/DosePrediction/predictions`），直接评估：

```bash
python - <<'PY'
from DosePrediction.Evaluate.evaluate_openKBP import get_Dose_score_and_DVH_score
pred_dir = "runs/DosePrediction/predictions"
gt_dir = "provided-data/nifti-test-pats"
dose, dvh, gt_stats, pred_stats, metric_dif = get_Dose_score_and_DVH_score(pred_dir, gt_dir)
print("Dose score:", dose)
print("DVH score:", dvh)
print("Per-metric mean absolute diff:", metric_dif)
PY
```

---

## 5. 第二步：运行 DVH 自动调参（推荐先短训）

脚本功能：
- 网格搜索 `hotspot_weight/coldspot_weight` 与 quantile；
- 每组参数都进行训练+验证；
- 输出 `dose_score`, `dvh_score` 以及 `D1/D95/D99/D_0.1_cc/mean` MAE；
- 结果写入 `dvh_tuning_results.json`。

### 快速调参（建议先跑）

```bash
python scripts/tune_dvh_hparams.py \
  --max-epochs 20 \
  --hotspot-weights "0.5,0.75,1.0" \
  --coldspot-weights "0.2,0.35,0.5" \
  --hotspot-quantiles "0.98,0.99" \
  --coldspot-quantiles "0.05,0.10" \
  --dose-weight 0.5 \
  --output-root runs/DosePrediction/dvh_tuning_stage1
```

### 第二轮精调（围绕第一轮最优点）

```bash
python scripts/tune_dvh_hparams.py \
  --max-epochs 35 \
  --hotspot-weights "0.85,1.0,1.15" \
  --coldspot-weights "0.25,0.35,0.45" \
  --hotspot-quantiles "0.985,0.99" \
  --coldspot-quantiles "0.08,0.10,0.12" \
  --dose-weight 0.5 \
  --output-root runs/DosePrediction/dvh_tuning_stage2
```

查看最优参数：

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("runs/DosePrediction/dvh_tuning_stage2/dvh_tuning_results.json")
data = json.loads(p.read_text(encoding="utf-8"))
print("Best config:")
print(json.dumps(data["best"], indent=2, ensure_ascii=False))
PY
```

### Windows（cmd）一键执行

```bat
scripts\run_dvh_optimization_windows.bat
```

---

## 6. 第三步：用最优参数做完整训练

> 下面命令按“先冻结、再解冻微调”给出，通常比单段训练更稳。

### 6.1 冻结阶段（主训练）

```bash
python - <<'PY'
from DosePrediction.Train.train_light_pyfer import main
main(freeze=True, delta1=10, delta2=8, max_epochs=300, fast_dev_run=False)
PY
```

### 6.2 解冻阶段（精调）

```bash
python - <<'PY'
from DosePrediction.Train.train_light_pyfer import main
main(freeze=False, delta1=10, delta2=8, max_epochs=120, fast_dev_run=False)
PY
```

---

## 7. 训练后推理与评估（统一口径）

```bash
python - <<'PY'
from DosePrediction.Train.train_light_pyfer import Pyfer, TestOpenKBPDataModule
import DosePrediction.Train.config as config
import pytorch_lightning as pl
from DosePrediction.utils.runtime import get_lightning_accelerator

cfg = {
    "act": "mish",
    "multiS_conv": True,
    "lr": 0.0006130697604327541,
    "weight_decay": 0.00016303111017674179,
    "delta1": 10,
    "delta2": 8,
    "hotspot_weight": 1.0,
    "hotspot_quantile": 0.99,
    "coldspot_weight": 0.35,
    "coldspot_quantile": 0.10,
}

model = Pyfer(cfg, freeze=False)
data = TestOpenKBPDataModule()
accelerator, devices = get_lightning_accelerator()
trainer = pl.Trainer(accelerator=accelerator, devices=devices, logger=False, enable_checkpointing=False)
trainer.test(model, datamodule=data)
print("Mean Dose score:", sum(model.list_dose_metric)/len(model.list_dose_metric))
print("Mean DVH score:", sum(model.list_DVH_dif)/len(model.list_DVH_dif))
PY
```

## 7.5 自动注入 best 参数并启动正式训练（一键）

当 `runs/DosePrediction/dvh_tuning_stage2/dvh_tuning_results.json` 已产生后，直接执行：

```bash
python scripts/train_with_best_dvh.py \
  --best-json runs/DosePrediction/dvh_tuning_stage2/dvh_tuning_results.json \
  --freeze-epochs 300 \
  --finetune-epochs 120 \
  --resume \
  --output-dir runs/DosePrediction/final_from_best_dvh
```

如果你只想跑冻结阶段（不做解冻微调）：

```bash
python scripts/train_with_best_dvh.py \
  --best-json runs/DosePrediction/dvh_tuning_stage2/dvh_tuning_results.json \
  --freeze-epochs 300 \
  --skip-finetune \
  --output-dir runs/DosePrediction/final_from_best_dvh
```

Windows（cmd）可直接运行：

```bat
scripts\train_with_best_dvh_windows.bat
```

> 如果 `dvh_tuning_results.json` 不存在，脚本会先自动调用调参流程，再启动正式训练。
> 如果你要“接着训练”，请确保加 `--resume`（Windows 一键脚本已默认带上）。

---

## 8. 提升效果的实操建议（按优先级）

1. **优先盯 `D_0.1_cc`**：先上调 `hotspot_weight` 或提高 `hotspot_quantile`（更聚焦顶端热点）。
2. **再压 `D99`**：适度增加 `coldspot_weight`，避免靶区低剂量冷点。
3. **不要一次改太多**：每轮只改 1~2 个维度，保留可解释性。
4. **统一评估口径**：固定同一验证集、同一脚本、同一后处理，否则分数不可比。

---

## 9. 常见坑位排查

- **预测全零或极低**：检查 `possible_dose_mask`、模型输出缩放（是否 `*70`）。
- **DVH 波动很大**：检查数据读取是否稳定（随机增强在验证集应关闭）。
- **调参结果不一致**：固定随机种子，固定 train/val 划分，减少并行扰动。
- **Windows 无法执行 `bash`**：使用 `scripts\run_dvh_optimization_windows.bat` 或在 PowerShell 里逐条运行 `python scripts/tune_dvh_hparams.py ...`。

---

如果你愿意，我可以下一步给你一版“**自动读取 best config 并发起正式训练**”的脚本，把上述步骤串成一个命令跑完。

## 附：Windows 下快速找最佳 epoch（不使用 heredoc）

在 Windows `cmd` 里不要用 `python - <<'PY'`。请直接执行：

```bat
python scripts\find_best_metrics.py --logs-root runs\logs\dose_prediction --metric mean_dose_score --mode max --checkpoint-dir runs\DosePrediction\final
```

> 说明：`mean_dose_score` 在训练日志中通常是负值（代码里做了取负），脚本会额外打印 `estimated_dose_score` 方便理解。
> 若出现 `[WARN] metrics step range and checkpoint step range do not overlap.`，说明你传入的日志目录和权重目录不是同一次训练。
> 新版脚本会自动给出 `[HINT] Possible matching checkpoint directories` 供你直接替换 `--checkpoint-dir`。

如果你要看最小 `val_loss`：

```bat
python scripts\find_best_metrics.py --logs-root runs\logs\dose_prediction --metric val_loss --mode min --checkpoint-dir runs\DosePrediction\final
```
