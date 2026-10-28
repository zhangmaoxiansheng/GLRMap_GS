# 数据准备：图片demo与历史批次

当前图片demo已由[`pipeline.demo`](../pipeline/demo.py)负责：复制RGB快照，新推理SAM3.1道路／标线／动态mask（不运行天空分割），再重新求解相机、重建道路、运行ReSplat及GS。它只需要100张666×370的JPEG，SfM、点云和GS都使用这100张图，命令与模型配置见[demo手册](../demo/README.md)。

以下契约仅适用于历史`pipeline.cloud/window`的1000帧批次入口；它仍需要原视频、已有mask与相机锚点，不是图片demo的准备步骤。

已有三环资产可按[Pipeline手册第1节](../pipeline/README.md#1-准备一个全新的点云工作目录)准备新的运行目录：只读链接原生RGB，独立复制mask与元数据，保留原始视频时间和SHA256，创建每批1000帧的图像链接。新SfM模型、数据库、900×500匹配图、点云和GS写入新目录，不回写资产。

| 准备项 | 契约 |
| --- | --- |
| 原生图像 | 10FPS、666×370，同一`crop=900:500:190:0`再缩小；帧名与元数据一致 |
| `frames_manifest.json` | 每帧`image/sha256/video_time_nominal`；提取时间不能替换为显示时间 |
| `masks/vehicle_valid` | 255=静态有效、0=动态忽略；原生画幅 |
| `masks/road` | 已精修语义道路mask；原生画幅，不是GS实际曲面覆盖mask |
| `sky_all/foreground_valid_hard` | DA-V2原生逆深度精确零值判天空；255=非天空，不膨胀／腐蚀 |
| 首批相机锚点 | 覆盖该批前100帧、对应相同裁剪与原生K的已知相机；本流程不提供无锚点新视频的初始标定 |
| 原长视频 | 可读，供`sfm.source`从确切时间重新提取900×500图；不插值放大已有JPEG |

SfM／RoMa使用动态有效与非天空交集；GS RGB只忽略动态，天空仍参与监督。道路适配根据曲面支持另外生成`covered_road_masks/`，不能与语义道路mask混用。相机求解属`sfm/`，适配后的K、特征和像素坐标由`sfm.source`同步维护。

当前已有资产适配：[数据与mask编排](../pipeline/cloud.py)、[分辨率适配](../sfm/source.py)。不要直接运行快照副本或靠修改其全局变量启动新实验。
