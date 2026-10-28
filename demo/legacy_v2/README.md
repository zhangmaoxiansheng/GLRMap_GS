# 100张图片从头重建GS

本demo的场景输入只有100张RGB图片。运行时自行生成动态／道路／标线mask、估计相机与内参、重建RoMa点云并拟合道路、运行ReSplat，再完成1000＋1000步GS训练及前进／变道展示。**ReSplat是必经阶段，需要配置SDK及预训练权重。**

入口：[pipeline/demo.py](../../pipeline/demo.py)。历史的`demo/legacy_v1/config.json`仅对应旧的固定几何GS阶段，不是本demo的完整入口。

## 下载图片

已在本机生成纯图片包`dist/demo-images-v2.zip`，约6.0MiB，含100张图及`demo/IMAGES_SHA256SUMS`；SHA256为`814382e003340db4c75444c7ee21c34a77ff146ca1d846efea555e722443f1f8`。该新包尚未上传；取得此包后在项目根目录解压，并执行`(cd demo && sha256sum -c IMAGES_SHA256SUMS)`。下方旧网盘仍可只提取相同图片。

现有[素材v1网盘](https://cloud.tsinghua.edu.cn/f/328530a0cc8a473c8e58/?dl=1)包含图片及旧的中间产物；以下命令只提取图片，不需要其余内容。命令在项目根目录执行。

```bash
mkdir -p dist
curl -fL 'https://cloud.tsinghua.edu.cn/f/328530a0cc8a473c8e58/?dl=1' \
  -o dist/demo-assets-v1.zip
printf '%s  %s\n' \
  '378f1ca0f89b909b0754d314b538e833d2c58324d52c565b8b88d5ba98323a48' \
  'dist/demo-assets-v1.zip' | sha256sum -c -
unzip -n dist/demo-assets-v1.zip 'demo/input/images/*' -d .
```

图片为三环350–360秒的`003400.jpg`–`003499.jpg`，共100张，10FPS、666×370。使用图片已有裁剪和分辨率，不依赖原视频，不放大至900×500，不导入旧内参或旧相机。运行入口读取的场景输入路径仅为`--images`。

## 配置依赖

先按[项目README](../../README.md)准备COLMAP／GLOMAP、RoMaV2／SuperPoint／LightGlue及各Python环境；Gaussian后端须按[完整安装手册](../../patches/GAUSSIAN_BACKEND.md)固定官方版本、应用两个补丁、编译CUDA扩展并通过`check_gs_backend --cuda`；按[ReSplat配置](../../README.md#resplat依赖与配置)克隆固定revision、应用补丁、下载16视角权重并完成导入检查。

图片demo还需要：

| `paths.local.json`配置 | 内容 |
| --- | --- |
| `python_sam` | SAM3.1推理的Python；已测Python3.12、Torch2.7.1+cu118 |
| `sam3_source` | [SAM3源码](https://github.com/facebookresearch/sam3)，本机revision `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7` |
| `sam3_checkpoint` | SAM3.1的`sam3.1_multiplex.pt`，按上游说明获取及安装模型依赖 |
| `python_sfm` | 含pycolmap的SfM环境 |
| `python_matching` | RoMaV2、LightGlue及pycolmap环境 |
| `python_resplat` | 完整ReSplat推理环境 |
| `python_gs` | 本项目CUDA rasterizer与simple-knn训练环境 |

配置模板见[paths.example.json](../../paths.example.json)。在五个Python环境分别执行：

```bash
/path/to/env/bin/python -m pip install --no-deps --no-build-isolation -e .
```

SAM环境另需NumPy、Pillow、OpenCV及SciPy；本次使用的补充依赖为：

```bash
/path/to/sam/python -m pip install opencv-python-headless==4.11.0.86 scipy==1.15.3
```

模型权重是软件依赖，不包含场景相机、mask或几何。配置示例中的`/opt/...`需要改为实际路径。

## 从图片执行

在已配置的matching或ReSplat环境中，从项目根目录执行：

```bash
python -m GLRMap_GS.pipeline.demo plan \
  --images demo/input/images \
  --output experiments/demo_images_no_sky

python -u -m GLRMap_GS.pipeline.demo run \
  --images demo/input/images \
  --output experiments/demo_images_no_sky
```

输出目录须不存在。中断后可用同一命令追加`--resume`，继续同一输入及同一实现的未完成阶段；不要修改已完成的阶段产物或在恢复期间换代码／模型。改变配置或算法时使用新输出目录。

流水线依次执行：

1. SAM3.1逐帧检测车辆、行人、道路及标线，沿用已有道路／标线合并方法。静态有效mask排除动态物体；不运行Depth Anything V2或任何天空分割。
2. SIFT顺序匹配及GLOMAP：共享未知内参的`PINHOLE`相机，优化内参，必须注册全部100帧并通过轨迹和观测检查。直接使用输出坐标及尺度，不做额外归一化。
3. 由新SfM共视关系选择180→360个图像对，RoMaV2双向查询、身份Snap、Track及固定相机三角化，每图20,000个源种子。沿用三环现有匹配／三角化筛选参数。
4. 从新点云拟合道路曲面，按像素射线增密，再生成道路高斯、约束及实际覆盖mask。
5. 新运行ReSplat：100帧中均匀选16个上下文，`dl3dv_16v_540x960`，精修2轮；将原生高斯的位置、旋转及SH转换到本次SfM世界坐标。
6. 训练1000＋1000步，固定道路全部参数，优化背景；RGB损失只忽略动态物体，天空参与监督；生成前进、左右横移和变道视频。

`gs_backend`除训练后端外，须保留已有的`complete_sanhuan_sam31_road_masks.py`，本入口直接使用其中的道路／标线合并函数。模型权重和后端源码不属于场景输入。

输入照片保持原生666×370；SfM、RoMa及GS均使用这一画幅与相应K，不读取原视频或放大成900×500。旧`shared/sky_masks/foreground_valid_hard`是匹配模块沿用的目录名，本demo写入其中的内容只是SAM动态有效mask，不含天空分割结果。

本次相机、点云和最终GS处于同一SfM世界坐标，但没有与历史三环批次对齐，也不声明米制尺度。道路曲面算法和既有SfM单位阈值保持原样；是否适合本次重建需由拟合验收及渲染检查确定。

## 输出与验收

```text
experiments/demo_images_no_sky/
├── images/                    # 输入图片实体快照
├── frames_manifest.json       # 逐图SHA256，时间为本片段0–10秒
├── masks/                     # 新推理的道路／标线／静态有效mask
├── sfm/                       # 新数据库、稀疏COLMAP模型、点云、相机审计
├── cameras.npz                # 新估计的world-to-camera外参及像素K
├── roma/                      # 新匹配、Track、三角化点云
├── dense_road/                # 完整融合点云及拟合道路
├── road_assets/               # 道路高斯与训练约束
├── resplat/                   # 本次新推理的原生高斯及视频
├── gs_config.json
├── gs/                        # 训练模型、统计、展示视频
├── progress.json              # 各阶段耗时与完成／失败状态
└── *.log                      # 分阶段完整日志
```

以`progress.json`中的`status=complete`及`execution_checks_passed=true`确认执行完成；同时核对`sfm/audit.json`全部100帧注册、`gs/training_report.json`的2000步和道路逐位冻结检查，以及`gs/visualization_complete.json`。画质还需查看新生成的前进与变道视频。

2026-10-28已从100张图片完整实跑，所有阶段完成。100/100帧注册，SfM平均重投影误差0.752px；ReSplat重新推理491,520个原生高斯；最终GS为1,235,094个点，其中852,563个道路高斯逐位固定。2000步训练约133秒，完整流程约9.3分钟（本机RTX4090）。

本次结果在`demo/result_images_no_sky/`：

- [前进变道](../result_images_no_sky/lane_change.mp4)
- [定点左右横移：初始化／训练后](../result_images_no_sky/lateral.mp4)
- [原图／ReSplat／混合初始化／训练后四栏对比](../result_images_no_sky/comparison.mp4)
- [训练后原轨迹](../result_images_no_sky/forward.mp4)
- [截图](../result_images_no_sky/preview.png)、[横移截图](../result_images_no_sky/lateral_preview.png)、[最终模型](../result_images_no_sky/gaussian.ply)、[完整统计](../result_images_no_sky/summary.json)

已抽查首／中／末帧和两处横移极值画面。车道线在抽查横移画面中可见，但路面仍有局部条带，训练后的天空云层出现块状／涂抹伪影，树木和建筑仍有重影；执行通过不等于画质验收通过。第一帧曲面支持只有405像素，未覆盖区域沿用可训练的ReSplat初始化。这里如实保留结果供比较，没有额外加入天空处理或几何修补。

纯图片包为`dist/demo-images-v2.zip`；本次结果包为`dist/demo-results-v2.zip`（约123.3MiB），SHA256为`4f9af7453348e05db9b19d2e1dbb210a81f57eeceba2133f192f3c2404c409ae`；解压后为独立的`demo-results-v2/`目录，可在该目录执行`sha256sum -c SHA256SUMS`。两者尚未上传网盘，旧v1网盘结果不是这一轮的输出。

## 历史参考结果v1

[结果v1网盘](https://cloud.tsinghua.edu.cn/f/3f08fe168a85469487c9/?dl=1)是此前读取预计算几何后训练的GS及视频，约122MiB，SHA256为`ebd3782911b29622d36545396437abf45b9fbb0bee6df03d81bd2676c3c3c094`。它用于历史画质参考，不作为图片全流程的输入，也不是本次从头重建的结果。旧素材v1中的`cameras/road/resplat/masks`同样不被新入口读取。
