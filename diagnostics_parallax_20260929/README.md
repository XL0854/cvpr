# RopStitch 输出后残余几何诊断

本目录独立于原始训练流程，研究以下候选问题：以冻结的 RopStitch `alpha=0.5` 输出为初始结果，利用两张 RGB 图像中的几何证据纠正残余局部错位，同时控制新增畸变、已对齐区域退化和有效内容损失。

这仍是诊断中的研究问题，不是已经成立的论文创新。

## 归因边界

- RopStitch 初始结果存在局部错位是输出层面的观察。
- 大残差筛选误伤正确对应属于后加筛选策略，不能归因于 RopStitch 内部。
- Adam-150 求解不足属于新增残差网格优化器，不能描述为 RopStitch 缺陷。
- AlphaPredictor、自适应 alpha、空间投影面和单独改进对应筛选均已暂停或归档；不训练筛选网络。
- ETH3D 深度和标定仅用于评价及诊断组 D，实际候选方案只读取两张 RGB。

当前定位见 [`CURRENT_RESEARCH_POSITION.md`](CURRENT_RESEARCH_POSITION.md)，最新统一强基线结论见 [`runs/unified_strong_baseline_review/REPORT.md`](runs/unified_strong_baseline_review/REPORT.md)。

## 已完成结论

在 courtyard、delivery_area 和 electro 共 15 个开发图对上：

- A：原始 RopStitch，固定 `alpha=0.5`；
- C-Adam：双向置信度、循环一致性和原 150 步 Adam；
- C-LBFGS：与 C 完全相同的对应、权重、13x13 网格和约束，仅充分求解；
- D-LBFGS：同一候选经 ETH3D 真值筛选，仅用于诊断。

点数加权几何误差为 A 38.73、C-Adam 31.10、C-LBFGS 24.09、D-LBFGS 22.64 px。C-LBFGS 在 13/15 对上改善 A，并解释约 91% 的 A 到 D 降幅。D 没有跨场景稳定领先，因此 C-LBFGS 只作为常规强基线保留，不包装成新方法。

目前保留的具体失败包括：

1. 结构可行边界阻塞，使 C/D-LBFGS 都无法接受更新；
2. 少量局部区域缺少被 C 接纳的正确空间支撑；
3. 更低几何误差可能伴随已对齐区域退化或有效覆盖下降。

## 目录

- `scripts/`：ETH3D 几何、固定候选、A/B/C/D、求解器与汇总脚本。
- `configs/`：冻结的三场景图对清单。
- `final_cross_scene_audit/`：对应筛选路线停止前的跨场景报告。
- `runs/residual_bottleneck_frozen3/`：三个代表样本的求解、约束和细网格诊断。
- `runs/unified_strong_baseline_review/`：最终 15 对统一强基线复核。
- `FROZEN_*.md`：实验运行前冻结的规则。
- `ARCHIVED_CORRESPONDENCE_SCREENING.md`：对应筛选路线的停止记录。

Git 仓库只保存代码、协议、汇总结果和关键图。ETH3D/UDIS-D 数据、模型权重、RoMa/DINO 权重、固定候选 NPZ 和完整中间形变结果需要在本地按路径准备，不提交到 Git。

## 关键入口

```bash
# 生成固定 RoMa 候选，需要本地 RoMa/DINO 权重和 GPU
CUDA_VISIBLE_DEVICES=1 bash diagnostics_parallax_20260929/RUN_GPU_FINAL_CANDIDATES.sh

# 统一 C/D L-BFGS 复核，复用已准备的初始网格和候选
python -B diagnostics_parallax_20260929/scripts/run_unified_strong_baseline.py

# 汇总、绘图及后处理计时
python -B diagnostics_parallax_20260929/scripts/summarize_unified_strong_baseline.py
```

运行前应先阅读相应 `FROZEN_*.md`，不得根据结果修改场景、图对、阈值或停止点。
