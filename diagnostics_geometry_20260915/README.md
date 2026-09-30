# RopStitch 几何蒸馏诊断（独立工作目录）

本目录仅用于诊断 A（外部几何匹配的可用性）与诊断 B（原 TPS 表达的逐样本拟合能力）。
不修改原始 `wCoefNet/`、`woCoefNet/` 源码，不训练或覆盖任何原网络权重。

## 结果入口

- `runs/pilot100/REPORT.md`：100 对正式诊断报告。
- `runs/pilot100/REVIEW.html`：逐对匹配抽样和优化前后拼接图，可直接浏览器打开。
- `runs/pilot100/summary.json`、`metrics.csv`：机器可读的汇总和逐对指标。
- `runs/pilot100/source_integrity.json`：与运行前的源码/权重 SHA-256 核对。
- `runs/geometry_checks.json`：TPS 数值、逆映射、梯度和留出点隔离检查。
- `runs/smoke2/`：2 对、10 步的流程校验，不混入正式统计。

## 本次协议

固定随机种子 20260915，50 对 UDIS training + 50 对 Classic development。
排除可访问的旧实验 manifest 中的图像哈希及 dHash 近重复；没有场景 ID，
不能保证场景独立，Classic 也不能视为全新未接触测试集。

原模型固定 α=0.5，Network/CoefNetwork 都置 eval 并冻结；权重严格加载。
RoMa outdoor 使用本地权重，560 粗分辨率 / 864 精细分辨率，不启用自定义相关性 CUDA 扩展。
双向 confidence ≥0.5、循环误差 ≤2px（512 坐标），8×8 空间均衡抽样。

留出约 25% 源图空间单元。拟合点与留出点、独立 SIFT 代理点在两图坐标中均间隔至少 8px。
SIFT 采用双向 ratio<0.7 与基础矩阵 MAGSAC；它是独立算法代理，不是人工真值。

每对固定优化 150 步，只修改两侧原有 13×13 TPS 网格的残差及匹配公共画布位置的辅助变量。
网络参数、单应变换、系数不变。网格位移每轴限制 ±20px，使用位移、平滑与折叠惩罚。
不根据留出集选步数或最优检查点。对应不足保留原结果，不排除出图像质量均值。

原 TPS 是逆向采样：本目录使用相同径向核和线性系统，在固定基线画布上计算。
对应点转移通过 Newton 求解逆映射，并检查残差。交换控制点只用作求根初值，
不把它冒充精确逆 TPS。数值实现与原 TPS 的坐标斜坡采样经过一致性验证。

## 解释边界

- 512×512 输入、固定基线画布、共同有效区指标是诊断口径，不是原论文原分辨率指标。
- teacher 留出点不等于真值；人工核验入口中所有条目初始为未核验。
- 无折叠仅指网格三角形和每单元 9 个 TPS 导数采样点未检测到折叠，不是全域证明。
- 原单应畸变项因单应与系数冻结而不变，不能据此宣称 TPS 自然性得到保证。
- 固定原重叠区域的覆盖惩罚是事后敏感性审查，不改变已完成拟合结果。
  丢失像素的 SSIM 置 0、归一化 MSE 置 1，是保守惩罚；不是标准 PSNR/SSIM。
- 逐样本拟合有收益不能推导训练后模型有收益、可跨域泛化或具备单次推理质量。

## 复现命令

在项目根目录执行；GPU 操作需要宿主机 GPU 访问权限。所有缓存重定向到本目录。

```bash
python -B diagnostics_geometry_20260915/scripts/preflight.py
python -B diagnostics_geometry_20260915/scripts/check_geometry.py
python -B diagnostics_geometry_20260915/scripts/run_diagnostics.py --run pilot100 --steps 150
python -B diagnostics_geometry_20260915/scripts/run_diagnostics.py --run pilot100 --audit_quality
python -B diagnostics_geometry_20260915/scripts/summarize.py --run pilot100
```

`run_diagnostics.py` 自动跳过已有结果。不要在已有 run 内改变参数重用缓存；新协议需使用新 run。
`prepare.py --run 新名字 --per_domain 50` 会创建新 manifest；已有 manifest 拒绝覆盖。
首次运行的协议与脚本哈希已保存于 `execution.json`。

`third_party/RoMa/` 为官方仓库 commit `77f8d68803526dcddfd9b7a46bc76125bdc25f15`。
额外依赖 loguru 0.7.3、einops 0.8.2 安装在 `third_party/python_deps/`，未改全局 Python 环境。
RoMa 与 DINOv2 权重保存在 `cache/torch/hub/checkpoints/`，大小和 SHA-256 见预检报告。
