# Research progress

更新日期：2026-09-30

## 研究定位

研究 RopStitch 输出后的 RGB 残余几何纠正。RopStitch 是冻结的初始化；候选匹配、筛选和残差网格优化均为后处理诊断。当前没有宣称新颖性。

## 已完成

1. 保留并归档 AlphaPredictor、自适应 alpha 搜索、空间投影面和五候选回退实验；不再作为论文主线。
2. 建立 ETH3D THIN_PRISM_FISHEYE 真值投影、遮挡/深度一致性检查和独立评价点划分。
3. 在 courtyard、delivery_area、electro 上固定 15 个图对，每对使用相同的 4096 个规则 RoMa 候选。
4. 验证正确但初始残差大的对应跨场景存在；按当前形变残差筛选会误伤这些点。
5. 完成 A/B/C/D 诊断并停止“单独改进对应筛选”路线：常规 C 已接近真值筛选 D，D 没有稳定综合优势。
6. 在三个代表样本上检查坐标、TPS 正反映射、反向采样和梯度；数值检查均通过。
7. 发现 Adam-150 的新增残差优化存在求解不足和结构可行边界问题，但这不是 RopStitch 内部缺陷。
8. 完成 15 对统一强基线：A、C-Adam、C-LBFGS、D-LBFGS，并计入 L-BFGS 强 Wolfe 线搜索的耗时和评价次数。

## 最终开发集结果

- A：38.73 px；
- C-Adam：31.10 px；
- C-LBFGS：24.09 px；
- D-LBFGS：22.64 px；
- 等 C-Adam 耗时预算：C-LBFGS 24.22 px，D-LBFGS 23.22 px。

C-LBFGS 获得约 91% 的完整 D 降幅、93.6% 的等耗时 D 降幅。返回网格无采样折叠，但最差有效重叠保留率约为 84%--85%，electro 的已对齐区域仍可能退化。

## 保留的具体失败

- `courtyard/DSC_0309_DSC_0311`：C/D-LBFGS 均无法接受结构可行更新，返回 A；C-Adam 仍有有限改善。
- `courtyard/DSC_0315_DSC_0317`：左下三个评价区域中，D 比 C 低 11--37 px；D 的最近正确支撑明显更近。
- `delivery_area/DSC_0695_DSC_0696`：D 比 C 低约 5 px，但差异无法仅由空间支撑解释。
- `delivery_area/DSC_0717_DSC_0718`：C 改善而 D 无可行更新，说明真值筛选并不稳定占优。

## 结论边界

现有15对都参与过诊断，只能作为开发证据。当前尚未找到足以直接支持新网络的稳定机制缺口。对应筛选和 alpha 方向继续归档。

## 关键材料

- `diagnostics_parallax_20260929/FROZEN_STRONG_BASELINE_REVIEW.md`
- `diagnostics_parallax_20260929/runs/unified_strong_baseline_review/REPORT.md`
- `diagnostics_parallax_20260929/runs/unified_strong_baseline_review/all_groups.csv`
- `diagnostics_parallax_20260929/runs/unified_strong_baseline_review/timing_per_pair.csv`
- `diagnostics_parallax_20260929/runs/unified_strong_baseline_review/failure_region_maps.jpg`
- `diagnostics_parallax_20260929/runs/residual_bottleneck_frozen3/REPORT.md`
- `diagnostics_parallax_20260929/final_cross_scene_audit/FINAL_REPORT.md`

## 下一步决策

先做聚焦文献查新：结构保持的局部非刚性拼接、稀疏/不均匀匹配支撑下的网格优化、可行域约束求解。如果这些问题已有充分常规解法，应更换切口。未经新证据，不训练筛选器、不增加模块、不扩数据。
