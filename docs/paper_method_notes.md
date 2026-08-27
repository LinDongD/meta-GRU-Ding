# 论文方法提取与复现决策

来源：Peng Ding, Minping Jia, Xiaoli Zhao, “Meta deep learning based rotating machinery health prognostics toward few-shot prognostics”, *Applied Soft Computing* 104 (2021) 107211, DOI: 10.1016/j.asoc.2021.107211。

## 术语表

| 统一术语 | 含义 | 本项目实现 |
|---|---|---|
| Meta-GRU | 基于模型无关元学习的 GRU 健康预测器 | `GRURegressor` + MAML 内外循环 |
| support set | 子任务内循环适配样本 | `Episode.support_x/y` |
| query set | 子任务外循环或最终评价样本 | `Episode.query_x/y` |
| TSUDA | 时间序列无监督域适配 | `SourcePreprocessor(coral_mmd)` 的可复现近似 |
| health degree | 机械健康度 | 单个轴承从 100 线性降至 0 |

## 论文到代码的证据映射

| 论文位置 | 论文信息 | 代码决策 |
|---|---|---|
| p.4, Fig. 2 | 原始振动、特征、UDA、support/query、元训练、少样本预测 | 完整流水线按此拆分 |
| pp.5-7, Eqs. 11-14 | GRU + MSE；support 内更新，query 跨任务更新 | `model.py`、`maml.py` |
| p.7, Algorithm 3 | 随机抽取任务；BPTT；内外双层梯度；目标域 support 微调 | `meta_step`、`adapt_parameters` |
| pp.7-8, Fig. 6 | 30 个统计特征、PCA、TSUDA、Meta-GRU | `features.py`、`preprocessing.py` |
| p.8 | 水平和垂直振动均参与分析；PCA 累计贡献率 99% | 默认拼接双通道，`pca_variance: 0.99` |
| p.9, Table 1 | 三个工况及对应 Bearing1/2/3_* | 按目录名前缀划分工况 |
| p.9 | 健康状态 100，最终失效 0 | `np.linspace(100, 0, count)` |
| p.9, Table A.3 | 100 epochs；α=β=1e-4；M=20；测试 support/query 各 9 对 | 作为论文基线，并允许搜索替换 |
| p.16-17 | 子任务越多通常精度更高，但时间和显存增加 | `meta_batch_size` 可搜索，完整复现可扩展到 20 |

## 算法概要

对源域的每个任务 $T_i$，从不相交数据中抽取 support 与 query。先在 support 上更新：

$$
\phi_i = \theta - \alpha \nabla_\theta \mathcal{L}_{T_i}^{support}(\theta).
$$

再用各任务 query 损失更新共享初始化：

$$
\theta \leftarrow \theta - \beta \nabla_\theta \sum_i \mathcal{L}_{T_i}^{query}(\phi_i).
$$

目标工况仅用少量 support 样本适配该初始化，query 只用于最终 MAE/RMSE 评价。

## 无法严格按原文复现的部分

论文没有公开 GRU 隐藏维度、层数、序列窗口、dropout、内循环步数和部分 TSUDA 优化细节，也没有公开作者代码。硬编码猜测值会制造伪复现。因此，本项目：

- 将网络容量和优化相关参数纳入源域验证搜索；
- 用均值匹配表示线性 MMD 项，用 CORAL 表示协方差项；
- 按论文的 $M=20$ 从合并源域随机生成子任务，并固定 9-shot support/query；
- 在 0–1 归一化健康度上使用可微 Adam 内循环，评价时还原为 0–100；
- 保留 `alignment: none` 便于验证域对齐贡献；
- 固定数据泄漏边界：预处理只拟合源域，超参数选择不看目标 query 标签。
