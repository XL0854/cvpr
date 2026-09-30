# 普通 RGB 残余网格网络（第一阶段）

本目录实现冻结 RopStitch 后的单侧 `13×13` 小幅残余纠正。它不修改
RopStitch/CoefNetwork，不预测对应可靠性，也不恢复 alpha 搜索。

## 坐标与输入

- 参考侧网格固定，只修改目标侧网格。
- 网格及几何损失使用 RopStitch 的 512 输入像素尺度。
- 输入为固定初始共同画布上的参考 RGB、目标 RGB、绝对 RGB 差及两个有效掩码，共 11 通道；网络内部另计算无参数的局部 RGB 相关期望位移与峰值 3 通道。
- 输出为目标初始网格上的 `169×2` 位移，最大分量由 `tanh` 限制；最终层零初始化。
- 深度和标定只在 `build_cache.py` 中离线生成监督；模型输入不包含深度、标定或真值。

## 监督与损失

ETH3D 可见性检查排除无效深度、遮挡、深度不一致和不确定深度边界。源图
8×8 单元中 `cell_id % 4 == 0` 的点只用于评价。其余点训练几何误差。
损失还包括：初始误差不超过 3 px 点的最终几何退化铰链、均匀一阶平滑、
位移幅度、三角形/TPS 折叠屏障，以及保持初始重叠采样仍位于目标有效域的损失。

## 当前数据边界

`development_split.json` 按场景使用 courtyard / delivery_area / electro，但这些
场景的 15 对都参与过前期诊断，因此只能验证实现和提供开发证据，不能称为
独立最终测试。正式最小划分仍需新增未使用场景，例如 kicker 作验证、terrace
作锁定测试；场景角色必须在训练前固定。

## 运行

```bash
python -B residual_learning_20260930/build_cache.py \
  --split residual_learning_20260930/configs/development_split.json \
  --output residual_learning_20260930/cache/development

python -B residual_learning_20260930/train_plain.py \
  --cache residual_learning_20260930/cache/development \
  --output residual_learning_20260930/runs/plain_development \
  --device cpu --epochs 80
```

CPU 运行只用于实现/开发检查，不作为最终推理耗时。GPU 驱动恢复后应在固定
设备上分别计量 RopStitch 前向、残余网络和最终采样时间。
