# 100张图片，从头运行完整GS demo

本demo的场景输入只有100张RGB图片。动态／道路mask、SfM相机、RoMa点云、道路曲面、ReSplat高斯和最终GS都重新生成。入口为[`pipeline/demo.py`](../pipeline/demo.py)。

固定道路高斯的位置、旋转与尺寸，训练SH颜色和透明度；背景正常训练。天空RGB参与监督，不运行天空分割。保留SfM原始坐标和尺度，不导入旧相机、点云或位姿锚点，不进行额外尺度归一化。

## 1. 准备100张图片

输入为`003400.jpg`–`003499.jpg`，三环350–360秒，10FPS、666×370，保持原裁剪。SfM、点云重建和GS都仅使用这100张图。运行不需要原视频或1000帧上下文。

新的素材包为`dist/demo-images-v4.zip`，仅包含100张JPEG和逐文件校验表。取得文件后，在项目根目录执行：

```bash
unzip -n dist/demo-images-v4.zip -d .
(cd demo && sha256sum -c IMAGES_SHA256SUMS)
```

图片与已有`demo-images-v2.zip`相同，可直接复用。当前[素材v1网盘](https://cloud.tsinghua.edu.cn/f/328530a0cc8a473c8e58/?dl=1)也包含这100张图；只提取图片即可，旧相机／点云／mask不会作为输入：

```bash
mkdir -p dist
curl -fL 'https://cloud.tsinghua.edu.cn/f/328530a0cc8a473c8e58/?dl=1' \
  -o dist/demo-assets-v1.zip
printf '%s  %s\n' \
  '378f1ca0f89b909b0754d314b538e833d2c58324d52c565b8b88d5ba98323a48' \
  'dist/demo-assets-v1.zip' | sha256sum -c -
unzip -n dist/demo-assets-v1.zip 'demo/input/images/*' -d .
(cd demo && sha256sum -c IMAGES_SHA256SUMS)
```

## 2. 安装依赖与配置

本仓库包含流程代码及必需补丁；第三方SDK和预训练权重需单独安装。安装顺序为：

1. 按[项目安装说明](../README.md#安装与本机配置)配置COLMAP／GLOMAP、RoMaV2／DINOv3／SuperPoint-LightGlue及Python环境，安装FFmpeg、ffprobe。
2. 按[Gaussian后端手册](../patches/GAUSSIAN_BACKEND.md)克隆固定revision，应用`gaussian-splatting.patch`和`diff-gaussian-rasterization.patch`，在GS环境重新编译CUDA扩展，通过`python -m GLRMap_GS.tools.check_gs_backend --cuda`。
3. 按[ReSplat手册](../README.md#resplat依赖与配置)安装固定revision、应用`resplat-infer-colmap.patch`并下载16视角权重。**ReSplat是必经阶段**。
4. 安装[SAM3](https://github.com/facebookresearch/sam3)及SAM3.1 multiplex权重。已测源码revision为`660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`，Python3.12、Torch2.7.1+cu118；SAM环境另需Pillow、NumPy、OpenCV与SciPy（已测`opencv-python-headless==4.11.0.86 scipy==1.15.3`）。
5. 将[paths.example.json](../paths.example.json)复制为`paths.local.json`，填写下列路径；五个环境分别安装本包。

```bash
/path/to/environment/bin/python -m pip install --no-deps --no-build-isolation -e /path/to/GLRMap_GS
```

| 配置项 | 内容 |
| --- | --- |
| `gs_backend`, `python_gs` | 应用两个补丁后的独立Gaussian后端及CUDA训练环境 |
| `resplat`, `resplat_checkpoint`, `python_resplat` | ReSplat SDK、16视角权重及推理／渲染环境 |
| `colmap`, `glomap`, `python_sfm` | 可执行文件及含pycolmap的SfM环境 |
| `romav2_source`, `dinov3_source`, `roma_assets`, `sparse_weights`, `python_matching` | 匹配源码、`models/romav2.0.1.pt`所在资产根、稀疏特征权重及匹配环境 |
| `sam3_source`, `sam3_checkpoint`, `python_sam` | SAM源码、`sam3.1_multiplex.pt`及环境 |
| `gpu_lock` | 同GPU任务共用的锁文件路径 |

软件依赖及模型权重不包含本场景的相机或几何。后端所需道路mask合并函数已包含在随库补丁中，不依赖旧实验脚本。不要复制旧环境的CUDA `.so`；在目标机器编译。

### 在新机器上准备SfM和匹配环境

COLMAP可执行程序使用3.9.1（CUDA特征提取），按[官方构建说明](https://github.com/colmap/colmap/tree/3.9.1)安装。GLOMAP使用[固定revision](https://github.com/colmap/glomap/tree/99806d0869f802fad218516a2e027793e7ca687d)，按该版本README安装构建依赖，再运行`cmake -S . -B build -GNinja`和`cmake --build build`。将实际二进制路径写入配置；安装`pycolmap`不能代替这两个可执行程序。

SfM Python环境可以独立创建：

```bash
conda create -n glrmap_sfm python=3.10 -y
conda activate glrmap_sfm
python -m pip install 'setuptools>=64' numpy==1.26.4 scipy==1.15.3 Pillow==12.0.0 pycolmap==4.2.0
python -m pip install --no-deps --no-build-isolation -e /path/to/GLRMap_GS
```

匹配环境采用已实跑的Torch／CUDA组合，RoMa源码由本入口通过`romav2_source`直接导入：

```bash
conda create -n glrmap_matching python=3.10 -y
conda activate glrmap_matching
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install 'setuptools>=64' numpy==1.26.4 scipy==1.15.3 Pillow==12.0.0 \
  opencv-python==4.11.0.86 pycolmap==4.1.0 einops==0.8.1 rich==14.2.0 \
  tqdm==4.67.1 kornia==0.8.2 matplotlib==3.10.3
python -m pip install --no-deps 'git+https://github.com/cvg/LightGlue.git@c2f8561576f0160b2cdaa3e4634b6b61d6cf6cb8'
python -m pip install --no-deps --no-build-isolation -e /path/to/GLRMap_GS

git clone https://github.com/Parskatt/RoMaV2.git /path/to/RoMaV2
git -C /path/to/RoMaV2 checkout 95c9968145c8906b7b59383258e9f73b02853d89
git clone https://github.com/facebookresearch/dinov3.git /path/to/dinov3
git -C /path/to/dinov3 checkout adc254450203739c8149213a7a69d8d905b4fcfa
```

以下权重需在运行前准备好；入口不在中途联网补下载：

| 权重 | 下载来源与放置位置 |
| --- | --- |
| RoMaV2 | [romav2.0.1.pt](https://github.com/Parskatt/RoMaV2/releases/download/v2.0.1/romav2.0.1.pt)，放到`<roma_assets>/models/romav2.0.1.pt`；SHA256见[`config/romav2.json`](../config/romav2.json)，其中已包含DINO模型参数 |
| SuperPoint | [superpoint_v1.pth](https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth)，放到`<sparse_weights>/superpoint_v1.pth` |
| LightGlue | [superpoint_lightglue.pth](https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth)，保存为`<sparse_weights>/superpoint_lightglue_v0-1_arxiv.pth` |
| SAM3.1 | 按[官方发布说明](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/RELEASE_SAM3p1.md)取得访问权限，从[模型页](https://huggingface.co/facebook/sam3.1)获取`sam3.1_multiplex.pt`，配置`sam3_checkpoint` |
| ReSplat | 下载及完整SHA256见[根README](../README.md#2-下载16视角权重) |

SAM环境先安装与上表已测环境一致的Torch／torchvision，再在固定revision的SAM源码下执行`pip install -e .`并补齐前述OpenCV、SciPy。ReSplat和GS涉及CUDA扩展，分别按各自的完整手册构建。以上记录来自当前实际使用的依赖与源码；没有声称已在另一台全新操作系统上验证所有安装步骤。

## 3. 从头运行

在已配置的matching或ReSplat环境中，从项目根目录执行：

```bash
python -m GLRMap_GS.pipeline.demo plan \
  --images demo/input/images --output experiments/demo_100_images

python -u -m GLRMap_GS.pipeline.demo run \
  --images demo/input/images --output experiments/demo_100_images
```

`plan`检查图片数量、尺寸和连续帧名，不启动GPU。`run`要求新输出目录。入口读取`--images`下的全部100张JPEG；没有额外选窗或1000帧模式。

默认使用`paths.local.json`；也可在命令前设置`GLRMAP_GS_PATHS=/absolute/path/paths.json`，所有子进程继承配置。仅在输入、代码和环境不变时用同一命令追加`--resume`，跳过已完成阶段。已部分写入的失败阶段不会自动清除；改变实现或输入时使用新输出目录。

## 4. 实际流程

| 阶段 | 本demo的设置 |
| --- | --- |
| SAM3.1 | 逐帧车辆／行人／道路／标线，动态膨胀10px，合并主路与附近标线；不分割天空 |
| SfM | COLMAP SIFT和顺序匹配，GLOMAP共享`PINHOLE`相机，估计内参、固定主点；100帧必须全部注册并通过检查 |
| RoMa | 100帧共视选对180→360，20,000种子／图，双向最多40,000查询／对；沿用已有Snap、Track与几何筛选 |
| 道路 | 固定SfM相机三角化，拟合曲面、像素射线增密；生成道路高斯、深度及实际覆盖mask |
| 渲染兼容检查 | 检查道路监督深度与固定后端`camera-Z > 0.2`要求一致；不兼容就停止，不自动缩放几何或修改裁剪阈值 |
| ReSplat | 100帧中均匀16上下文，`dl3dv_16v_540x960`，精修2轮，重新推理并转换到本次SfM坐标 |
| GS | 1000＋1000步，每视角每阶段10次；`road_train_appearance=true`，道路几何固定、SH与透明度可训练，背景正常训练 |
| 展示 | 原轨迹、定点左右横移、前进变道、原图／ReSplat／初始化／训练后四栏视频 |

RGB损失仅忽略动态物体，天空继续参与监督。道路逆深度权重为第一阶段0.5→0.05、第二阶段0.05。不进行高斯增密或剪枝；曲面未覆盖的ReSplat道路仍可训练。

所有图像和K保持666×370对应关系，不放大为900×500。历史目录名`shared/sky_masks/foreground_valid_hard`中写入的仅是SAM动态有效mask，不能据名称理解为运行了天空分割。

## 5. 输出与验收

```text
experiments/demo_100_images/
├── images/                    # 100张实体输入快照
├── frames_manifest.json       # 逐图SHA256和0–10秒相对时间
├── masks/                     # 新生成的动态／道路／标线mask
├── sfm/                       # 新COLMAP数据库、稀疏模型、点云及相机审计
├── cameras.npz                # 本次100帧world-to-camera外参和像素K
├── roma/B1_shared/            # RoMa三角化完整点云及Track
├── dense_road/FUSED_AFTER/    # 原非道路点＋拟合增密道路
├── road_assets/               # 道路高斯、深度、覆盖mask和renderer_audit.json
├── resplat/                   # 新推理的原生高斯与视频
├── gs_config.json
├── gs/                        # 训练模型、审计、视频及截图
├── demo_source.py             # 本次入口源码快照
└── plan.json / progress.json / *.log
```

核对`progress.json`执行完成、SfM注册100帧、`renderer_audit.json`通过、`gs/training_report.json`完成2000步且`road_geometry_bit_exact=true`，以及`visualization_complete.json`完成。道路外观发生更新时，`road_fields_bit_exact=false`符合当前设置。

本次模型只使用自己的SfM坐标，不自动对齐历史三环全局模型，也不声明米制尺度。执行检查通过与画质验收分别记录；查看连续视频和横移极值后再判断效果。

## 6. 查看结果与交付

2026-10-28已从100张图片完整实跑：100/100帧注册，平均重投影误差0.695px；新RoMa三角化952,264点，完整融合点云1,601,201点；ReSplat重新推理491,520个原生高斯，最终GS为1,305,705个高斯。其中926,407个道路高斯的几何逐位不变、外观参与训练。道路监督深度最小0.2646，近裁剪不兼容像素为0。完整流程约9.4分钟，GS训练约138秒，硬件为RTX4090。

本次使用独立的官方固定版本＋随库补丁后端完成实跑；92项CPU检查通过，安装包已在隔离导入环境验证入口和三份补丁完整。未在另一台全新操作系统上重新安装所有SDK。

- [前进变道](result_100_images/videos/lane_change.mp4)、[定点左右横移](result_100_images/videos/lateral.mp4)
- [四栏过程对比](result_100_images/videos/comparison.mp4)、[原轨迹](result_100_images/videos/forward.mp4)
- [原视角PNG](result_100_images/preview.png)、[左侧极值PNG](result_100_images/lateral_preview.png)、[右侧极值PNG](result_100_images/lateral_right.png)
- [最终GS](result_100_images/GS/gaussian.ply)、[完整点云](result_100_images/point_cloud/full.ply)、[COLMAP（含100张图）](result_100_images/colmap/)
- [结果说明](result_100_images/README.md)、[实际统计与检查](result_100_images/summary.json)

已抽查原视角0／50／99和两处横移极值，车道线可见、初始化道路条带减轻；天空仍有碎块，树木和建筑边缘仍有重影，局部道路纹理仍有条带。视频与模型均保留实际输出，没有以旧模型替换；当前记录`visual_quality_accepted=false`，不承诺等同于旧长序列画质或数值逐位重现。

交付文件均在`dist/`：`glrmap-gs-demo-source-v4.zip`为代码、文档、三份补丁；`demo-images-v4.zip`为100张输入图；`demo-results-v4.zip`为上述GS、完整点云、COLMAP及图片、视频和结果记录。结果用于比较，不作为从头运行的输入。新包尚未上传网盘，旧网盘链接不是这一轮结果。

源码包解压为`GLRMap_GS/`；在此目录解压图片包。结果包解压为独立`demo-results-v4/`，可直接查看视频，执行`(cd demo-results-v4 && sha256sum -c SHA256SUMS)`核验完整性。需要README中的相对结果链接时，将结果目录放到项目的`demo/result_100_images/`。各ZIP的SHA256另附在同名`.zip.sha256`和`dist/demo-v4-packages.json`中。

本地旧版目录与ZIP保留供追溯；当前入口只读`input/images/`。旧v1网盘结果来自预计算长序列几何，v2为道路全部冻结的100帧流程，v3为未通过画质验收的1000帧尝试；它们均不是本次参考结果。
