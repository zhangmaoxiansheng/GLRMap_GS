# GLRMap_GS

道路场景的 RoMaV2 点云重建、道路曲面增密与 Gaussian Splatting 流水线。

项目已从 `superDIM/code_review_sanhuan` 迁出，独立包名为 `GLRMap_GS`，命令入口为 `python -m GLRMap_GS`（或 `glrmap-gs`）。源码不再依赖旧目录层级。GS 后端、ReSplat、匹配模型和运行环境仍作为外部依赖配置。

## 完整流程

已有10FPS图像、动态／道路／天空mask、原视频和相机锚点 → GLOMAP → 几何选对 → RoMaV2＋SIFT／SuperPoint-LightGlue Snap → 共享Track与固定相机三角化 → 道路曲面拟合和射线增密 → ReSplat背景初始化＋道路高斯 → 固定道路、训练背景 → PNG、原轨迹和左右变道MP4、PLY。

单次处理20–100帧窗口；SfM可使用1000帧上下文，匹配仅计算涉及目标窗口的对。GS两阶段各按每视图10步：20帧共400步，100帧共2000步。默认不启用Pi3X／DA3支线。完整输入契约、逐阶段命令和输出说明见 **[Pipeline手册](pipeline/README.md)**。

## 安装与本机配置

当前采用Linux、NVIDIA GPU、多个现有Python环境，避免混用各模型的Torch／CUDA依赖。需要FFmpeg、ffprobe；后台执行可用tmux。

```bash
cd /path/to/GLRMap_GS
cp paths.example.json paths.local.json
# 编辑 paths.local.json，填写本机项目、权重、可执行程序和Python路径。
# 在SfM、匹配、ReSplat、GS四个环境中分别执行：
/path/to/environment/bin/python -m pip install --no-deps --no-build-isolation -e .
```

`pip install`仅注册本项目，不安装GPU模型依赖。四个环境均需已有setuptools≥64。激活匹配环境后启动主入口；其它环境由流水线调用。可选Pi3X／DA3环境也需安装本包。

配置见 [paths.example.json](paths.example.json)。默认读取包根目录的 `paths.local.json`，或使用环境变量 `GLRMAP_GS_PATHS=/absolute/path/paths.json`。配置中的相对路径以包根目录解析；输入、输出建议使用绝对路径。配置不进入Git。迁移机器后修改配置，不需要修改算法代码。

## 一键运行

以下输入是已准备好的批次，目录结构见Pipeline手册。`stop-frame`为不包含的结束帧。

```bash
python -m GLRMap_GS plan \
  --input /data/sanhuan/cloud --output /data/results/window_394_396 \
  --first-frame 3840 --stop-frame 3860 --reuse-sfm

python -m GLRMap_GS run \
  --input /data/sanhuan/cloud --output /data/results/window_394_396 \
  --first-frame 3840 --stop-frame 3860 --reuse-sfm
```

`plan`不启动计算。`--reuse-sfm`显式复用并检查输入相机审计；没有已完成SfM时去掉该参数。输出必须是新目录，阶段失败即停止。小窗口不会自动扩成整批三角化或训练。

这套输入仍约定三环的10FPS、900×500中心crop及666×370监督图／对应固定针孔K，**不是任意视频直接运行的通用入口**。原视频抽帧、SAM／DA-V2 mask准备尚未封装；新相机先检查并适配输入约定。

主要结果：`dense_road/comparison/road_bev_side.png`、`gs/review/*.png`、`gs/forward_lane_change.mp4`、`gs/trained_forward.mp4`、`gs/trained/point_cloud/iteration_400/point_cloud.ply`（20帧）。详见手册输出表。

## 功能归属

| 目录 | 当前职责与入口 |
| --- | --- |
| [data_preparation/](data_preparation/README.md) | 原视频抽帧／SAM／天空推理尚未迁入；已有资产准备方法见Pipeline第1节 |
| [sfm/](sfm/README.md) | GLOMAP、固定重叠锚点及相机；900×500图像／K／SIFT同步适配 |
| [matching/](matching/README.md) | 共视选对、RoMaV2双向查询、SIFT／SuperPoint＋LightGlue及身份Snap |
| [triangulation/](triangulation/README.md) | 按共享参考源点构建Track、固定Pose/K求XYZ、几何验收和分块合并 |
| [road/](road/README.md) | 道路Track提取、受支持曲面拟合、射线增密、完整非道路保留；不含专项车道线修复入口 |
| [depth_prior/](depth_prior/README.md) | 可选DA3／Pi3X相机条件化推理、深度约束、替代、Anchor及补点 |
| [gs/](gs/README.md) | 道路GS适配、原生ReSplat坐标／协方差／SH转换、初始化及固定道路训练 |
| [visualization/](visualization/README.md) | 单文件离线WebGL、完整点云投影、BEV、GS同轨迹／横移／变道和视频过渡 |
| [pipeline/](pipeline/README.md) | 1000帧点云队列；单窗GS与100帧／25帧重叠队列；环境、进度、恢复与产物交接 |

默认1000帧点云批次重叠100帧；三角化按100张参考图分块但保留全批邻居；GS窗口100帧、步长75、重叠25帧。不能把这三种单位混用。各模块通过包内导入运行，不通过试验脚本修改全局路径来复用算法。

## 迁移所需的项目与配置

外部项目保留独立目录，不自动复制进本仓库。各自的依赖与许可沿用来源项目。

| 配置项 | 依赖／约定 |
| --- | --- |
| `gs_backend` | 当前经过修改的Gaussian Splatting后端：`scene/`、`gaussian_renderer/`、`arguments/`、`utils/`及对应CUDA扩展。迁移时保留本地修改，不能用未适配的上游clone直接替代 |
| `colmap`, `glomap` | COLMAP 3.9.1 CUDA与GLOMAP；已测GLOMAP revision `99806d0869f802fad218516a2e027793e7ca687d` |
| `romav2_source` | RoMaV2源码；已测 `Parskatt/RoMaV2` revision `95c9968145c8906b7b59383258e9f73b02853d89` |
| `dinov3_source` | DINOv3本地hub源码；已测 `adc254450203739c8149213a7a69d8d905b4fcfa` |
| `roma_assets` | 包含 `models/romav2.0.1.pt` 的资产根目录；路径／SHA256见本库 `config/romav2.json`，权重必须在该资产根下。DINO参数已包含在此checkpoint |
| `sparse_weights` | 含 `superpoint_v1.pth`、`superpoint_lightglue_v0-1_arxiv.pth` 的目录；LightGlue代码需安装，已测revision `c2f8561576f0160b2cdaa3e4634b6b61d6cf6cb8` |
| `resplat` | 独立ReSplat源码及本地修改，含src/scripts/config；已测 `cvg/resplat` revision `cae7ddc4cdbd80e05e9f5fa00f5ea02c4e9056b1` |
| `resplat_checkpoint` | ReSplat16视角权重；默认 `resplat/pretrained/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth` |
| `gpu_lock` | 同GPU的任务共用同一锁路径；ReSplat推理等待至少18000MiB空闲显存 |
| `python_sfm/matching/resplat/gs` | 四个环境的Python绝对路径；当前主入口本身须在matching环境运行 |
| `pi3x_source`, `da3_source`, `da3_model` | 可选支线的SDK／DA3权重目录；Pi3X权重用推理命令的 `--ckpt` 指定 |

已验证的环境版本（不是完整锁文件）：

| 环境 | 关键版本 |
| --- | --- |
| SfM | Python3.10、Torch2.5.1+cu121、pycolmap4.2.0、NumPy1.26.4 |
| matching | Python3.10、Torch2.7.1+cu118、torchvision0.22.1、pycolmap4.1.0、NumPy1.26.4、SciPy1.15.3 |
| ReSplat | Python3.12、Torch2.7.1+cu118、gsplat1.5.3、Hydra1.3.2、plyfile1.1 |
| GS | Python3.9、Torch2.6.0+cu118、torchvision0.21.0、NumPy2.0.2、plyfile1.1.3；本地编译的rasterization和simple-knn |

换机器还需保存各环境的conda导出／pip freeze和外部项目本地修改；不要直接复制venv或不同Torch版本的CUDA `.so`。原机验证GPU为RTX4090。

数据单独携带：原视频、图像及哈希／时间manifest、各mask、首批相机锚点、matching_settings。只从GS继续时还需要道路曲面、来源像素、ReSplat原生PLY和相机等阶段输入。包含软链接的权重、数据及源码须同时迁移链接目标。历史实验的路径／哈希记录保持原样，不改写它们来绕过恢复检查。

## 检查

```bash
python -m GLRMap_GS --help
OPENBLAS_NUM_THREADS=1 python -m unittest \
  GLRMap_GS.triangulation.test_contracts GLRMap_GS.road.test_contracts \
  GLRMap_GS.depth_prior.test_contracts GLRMap_GS.gs.test_contracts \
  GLRMap_GS.visualization.test_contracts GLRMap_GS.pipeline.test_contracts -q
```

2026-10-26迁出后88项CPU检查通过，独立入口重跑394–396秒完整流程耗时133秒，400步GS及渲染均完成（显式复用已审计SfM，没有重跑1000帧SfM）。本机记录见 `experiments/sanhuan_394_396_migration_check/progress.json`。394–396秒20视图完整流程曾完成46对RoMa、231405个三角化点、125543个拟合道路点、400步训练及全部渲染；执行通过不代表其它场景画质自动达标，应查看PNG及左右外插视频。

`CORE_PIPELINE.md`与各功能README保留算法契约。迁移前的 `repository/`、`external_sources/`、`reference_configs/`、来源清单仅作为本地历史快照保留，已从Git交付范围排除；生产模块不导入它们。模型、实验、数据和本机配置也不进入Git。
