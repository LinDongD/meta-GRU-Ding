# Meta-GRU 复现：少样本轴承健康预测

本项目复现 Ding、Jia 与 Zhao 在 *Applied Soft Computing* 104 (2021) 107211 中提出的 Meta-GRU 核心流程，并允许通过源域内部验证自动选择论文未充分公开或不应固定的超参数。

## 实现范围

- 从 PRONOSTIA `acc_*.csv` 同时读取水平、垂直振动信号。
- 每个通道提取论文 Tables A.1-A.2 的 30 个时域/频域统计量；默认拼接为 60 维。
- 用源域拟合标准化和 PCA，累计解释方差为 99%。
- 用均值匹配（线性 MMD）与 CORAL 协方差匹配实现稳定的 TSUDA 近似。
- 将每个源域轴承视为一个任务，构造不相交的 support/query 集合。
- 用可微内循环和跨任务外循环实现一阶或二阶 MAML Meta-GRU。
- 在源域留出轴承上进行随机超参数搜索，再用全部源域训练，最后执行目标域 few-shot 适配与评估。

论文明确设置为：100 epochs、内外学习率均为 `1e-4`、20 个 subtasks、目标域 support/query 各 9 对样本。论文没有公开 GRU 隐藏层维度、层数和序列窗口等足够细节，所以默认配置把它们纳入搜索，而不是猜成一个“原文值”。

## 与论文的边界

论文 TSUDA 同时优化协方差距离和 MMD，并称采用 packing number 与黎曼梯度下降，但没有给出足以逐行复现的核函数、权重、流形约束及优化细节。本项目的 `coral_mmd` 对齐具有同一统计目标，但属于可审计、数值稳定的工程近似。若要做严格论文对照，可将 `alignment` 改为 `none` 形成消融基线。

健康度标签按每个轴承全寿命从 100 线性降至 0。超参数仅使用源域内部验证任务选择，不使用目标域 query 标签调参，以避免测试泄漏。

## 安装与运行

本项目已在 Conda 环境 `PTO_250_124` 验证：Python 3.9.25、PyTorch 2.5.0 + CUDA 12.4，RTX 4060 Ti 可用。VS Code 已固定使用 `D:\Anaconda\envs\PTO_250_124\python.exe`。

在 VS Code 中打开终端后运行：

```powershell
conda activate PTO_250_124
python -m pip install -e ".[dev]" --no-build-isolation
python -m meta_gru.check_env
pytest
python -m meta_gru.train --config configs/smoke.yaml --output-dir outputs/smoke
python -m meta_gru.train --config configs/search.yaml --output-dir outputs/c1_to_c2
```

改变 `source_condition` 与 `target_condition` 即可执行论文中的六种跨工况方向。首次运行会在 `artifacts/cache/` 生成特征缓存；该目录和训练输出均被 Git 忽略。

运行时每个 epoch 都会显示进度条、query MSE loss 和 RMSE。结果会同时写入：

| 文件 | 内容 |
|---|---|
| `metrics/epoch_metrics.csv` | 每个搜索 trial 和最终训练的逐 epoch loss、RMSE、梯度范数、学习率、耗时与 CUDA 峰值显存 |
| `metrics/search_trials.csv` | 每组超参数及源域验证 MAE |
| `metrics/target_trial_metrics.csv` | 每个目标轴承、每次 few-shot trial 的 MAE/RMSE |
| `metrics/target_query_predictions.csv` | 所有目标 query 点的真实 RUL、预测 RUL、残差和绝对误差 |
| `metrics/target_full_curves.csv` | 每个目标轴承的完整寿命预测曲线及 support 点标记 |
| `plots/epoch_loss_rmse.png` | 所有 epoch 的 loss/RMSE 曲线 |
| `plots/training_diagnostics.png` | 梯度范数、epoch 耗时和 CUDA 峰值显存 |
| `plots/hyperparameter_search_mae.png` | 超参数试验验证 MAE |
| `plots/target_rul_curves.png` | 各轴承估计 RUL 与真实 RUL 曲线 |
| `plots/rul_true_vs_estimated.png` | 真实值与预测值散点图 |
| `plots/rul_residuals_and_absolute_error.png` | 残差分布及全寿命绝对误差 |
| `plots/target_trial_mae_rmse.png` | 不同目标轴承多次试验的 MAE/RMSE 箱线图 |

CSV 使用 `utf-8-sig` 编码，可直接用 Excel、Python、Origin 或 MATLAB 读取并重新作图。模型参数保存为 `model.pt`，完整汇总保存为 `summary.json`。

其中 `smoke.yaml` 只运行极小规模训练，用于确认特征缓存、CUDA、训练、保存和绘图整条流水线；确认无误后再运行 `search.yaml`。如果要自行调整，可修改：

```yaml
search:
  trials: 2
  epochs_per_trial: 2
train:
  epochs: 3
  target_trials: 2
```

注意：首次运行仍需从全部 `acc_*.csv` 提取特征，耗时主要在磁盘读取；完成后会复用 `artifacts/cache/`，后续试验会快很多。


## 推荐实验顺序

1. 先把 `search.trials` 设为 2、`epochs_per_trial` 设为 2，验证整条流水线。
2. 再恢复配置进行完整搜索；CPU 会很慢，建议 CUDA GPU。
3. 分别运行 1→2、1→3、2→1、2→3、3→1、3→2。
4. 至少报告目标域多次 episode 的 MAE 均值与标准差，并与 `alignment: none`、普通 GRU 预训练/微调做消融对比。

论文方法提取和复现决策见 [docs/paper_method_notes.md](docs/paper_method_notes.md)。
