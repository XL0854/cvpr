# 缺失资源与下载清单

检查日期：2026-09-29。项目及 `/workspace/python_pro` 下没有发现 ETH3D 的 `cameras.txt`、`images.txt` 与公开深度文件。磁盘可用空间约 626 GB，容量不是阻塞；当前环境缺少 `7z` 解压命令。

官方入口：<https://www.eth3d.net/datasets>

ETH3D 官方说明确认：高分辨率训练集的 `_depth.7z` 是与**原始畸变 DSLR 图像**匹配的 float32 行主序深度缓冲；无效深度为正无穷。相机标定采用 COLMAP `cameras.txt` / `images.txt`，外参把世界坐标变换到相机坐标。不能直接把这些深度当作 undistorted 图的深度。

## 第一步：一场景端到端检查

建议先下载 courtyard，共约 0.8 GB 压缩体积：


- `https://www.eth3d.net/data/courtyard_dslr_jpg.7z`（约 0.4 GB）
- `https://www.eth3d.net/data/courtyard_dslr_depth.7z`（约 0.4 GB）

期望放置根目录：

```text
/workspace/python_pro/rop/RopStitch-main/data/ETH3D/
```

解压后必须同时找到：原始 DSLR JPG、`dslr_calibration_jpg/cameras.txt`、`dslr_calibration_jpg/images.txt` 和对应的 `ground_truth_depth` 文件。脚本在目录结构核实前不会猜测路径。

## 第二步：20—30对开发集合

建议只扩展到五个独立训练场景，总压缩体积约 4.0 GB：

| 场景 | 原始 JPG | 深度 | 合计 |
|---|---:|---:|---:|
| courtyard | 0.4 GB | 0.4 GB | 0.8 GB |
| delivery_area | 0.4 GB | 0.4 GB | 0.8 GB |
| electro | 0.4 GB | 0.6 GB | 1.0 GB |
| kicker | 0.4 GB | 0.4 GB | 0.8 GB |
| terrace | 0.2 GB | 0.4 GB | 0.6 GB |

对应 URL 统一为：

```text
https://www.eth3d.net/data/<scene>_dslr_jpg.7z
https://www.eth3d.net/data/<scene>_dslr_depth.7z
```

其中 `<scene>` 为 `courtyard`、`delivery_area`、`electro`、`kicker`、`terrace`。

`playground`、`relief`、`office` 等场景暂不用于开发参数选择，可留作后续独立测试。图对只能按共同可见真值覆盖、基线和视场重叠等预先规定的可评价条件选择，不能按方法最终误差挑选。

## 解压依赖

当前容器没有检测到 `7z`。Ubuntu 通常使用：

```bash
apt-get update && apt-get install -y p7zip-full
```

这是系统级安装命令，应由容器管理员执行。本诊断没有自动安装依赖或下载约 4 GB 数据。

