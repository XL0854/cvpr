# 后续检验启动等待诊断

2026-09-16 分步实测：Python 能启动；NumPy 在启动后约 5.48 秒完成导入，OpenCV 约 13.48 秒，PyTorch 约 56.21 秒。随后沙箱内 CUDA 检查返回 Error 304，is_available=False。

堆栈先停留在 torch._C 二进制扩展加载，随后推进到 Python 标准库和 torchvision 依赖加载。进程多次显示 D 状态；系统 /proc/pressure/io 当时 full avg10=34.87。说明存在明显的系统 I/O 等待，但尚未定位到具体磁盘、文件系统或后台负载原因，不能将它归因为算法死循环。

机器当时 MemAvailable 约 88 GiB，不能仅依据 free 内存较少或 swap 已满判定当前内存耗尽。

处理：给独立 check_safety.py 添加启动输出与 45 秒堆栈诊断；获准切换宿主 GPU 执行。不重装依赖、不修改原模型源码或参数。
