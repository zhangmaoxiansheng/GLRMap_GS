# SfM与相机

负责GLOMAP稀疏重建、相机Pose／内参、跨批重叠锚点及坐标系检查。输出供匹配、三角化和GS共用的相机与稀疏模型；不负责RoMa匹配。

整理后的实现：

- [reconstruction.py](reconstruction.py)：`run_block`显式接收输出、批次、参考相机和可执行文件路径；保留GLOMAP、固定重叠锚点BA和相机验收。
- [source.py](source.py)：`prepare_source`同步适配原生900×500图像、K、SIFT数据库与稀疏观测；保持Pose和XYZ。
- [geometry.py](geometry.py)：共享的相机读取、像素射线、极线误差和固定相机参数转换。

历史来源（不作运行入口）：

- [GLOMAP、重叠锚点与相机检查](../repository/run_exp142_sfm.py)
- [1000帧批次中的SfM调用](../pipeline/cloud.py)

相机与像素交接约定、调用限制和核对结果见[核心整理说明](../CORE_PIPELINE.md)。原始快照不改写。

首次使用见[Pipeline手册](../pipeline/README.md)，由`pipeline.cloud sfm/run`传入批次和参考相机；不需要修改本模块全局变量。

100张图片从头求相机的入口见[`pipeline.demo.sfm`](../pipeline/demo.py)：共享未知内参的PINHOLE，GLOMAP优化内参，不读取历史相机锚点；输出本次独立SfM坐标系与`cameras.npz`。图片保持原生666×370，不调用依赖原视频的`prepare_source`。
