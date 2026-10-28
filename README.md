# GLRMap_GS

道路场景的 RoMaV2 点云重建、道路曲面增密与 Gaussian Splatting 流水线。

项目已从 `superDIM/code_review_sanhuan` 迁出，独立包名为 `GLRMap_GS`。图片demo入口为 `python -m GLRMap_GS.pipeline.demo`；历史预准备批次入口为 `python -m GLRMap_GS`（或 `glrmap-gs`）。源码不再依赖旧目录层级。GS 后端、ReSplat、匹配模型和运行环境仍作为外部依赖配置。

## 100张图片的完整demo

[`demo/`](demo/README.md)只输入三环350–360秒的100张RGB图片（`003400.jpg`–`003499.jpg`，10FPS、666×370），从头完成以下流程：

```text
100张RGB → SAM动态／道路／标线mask → COLMAP特征与匹配＋GLOMAP相机
        → RoMaV2＋SIFT／SuperPoint-LightGlue Snap → 固定相机三角化
        → 道路曲面拟合、射线增密 → ReSplat重新推理 → 道路＋背景GS初始化
        → 1000＋1000步GS训练 → 最终PLY、原轨迹／横移／变道视频
```

同一100张图用于SfM、点云与GS。道路位置／旋转／尺寸固定，SH颜色和透明度参与训练；背景正常训练。天空RGB参与损失，不运行天空分割。保留新SfM坐标及尺度，不导入旧相机锚点、不做额外尺度归一化。

配置完下述依赖后，在项目根目录运行：

```bash
python -m GLRMap_GS.pipeline.demo plan --images demo/input/images
python -u -m GLRMap_GS.pipeline.demo run --images demo/input/images \
  --output experiments/demo_100_images
```

素材下载、输出目录、实跑结果及打包文件见[demo手册](demo/README.md)。2026-10-28已从100张图片完整实跑，2000步训练与全部渲染完成；[变道视频](demo/result_100_images/videos/lane_change.mp4)和画质限制见demo手册。代码与补丁在本仓库；第三方SDK和预训练权重按路径配置。运行不需要旧实验目录、原视频或额外900张图片。当前`demo`入口与下文历史`pipeline.window/cloud`批次入口的输入约定不同。

## 安装与本机配置

当前采用Linux、NVIDIA GPU、多个现有Python环境，避免混用各模型的Torch／CUDA依赖。需要FFmpeg、ffprobe；后台执行可用tmux。

```bash
cd /path/to/GLRMap_GS
cp paths.example.json paths.local.json
# 编辑 paths.local.json，填写本机项目、权重、可执行程序和Python路径。
# 在SAM、SfM、匹配、ReSplat、GS五个环境中分别执行：
/path/to/environment/bin/python -m pip install --no-deps --no-build-isolation -e .
```

`pip install`仅注册本项目，不安装GPU模型依赖。五个环境均需已有setuptools≥64。激活匹配环境后启动主入口；其它环境由流水线调用。可选Pi3X／DA3环境也需安装本包。

配置见 [paths.example.json](paths.example.json)。默认读取包根目录的 `paths.local.json`，或使用环境变量 `GLRMAP_GS_PATHS=/absolute/path/paths.json`。配置中的相对路径以包根目录解析；输入、输出建议使用绝对路径。配置不进入Git。迁移机器后修改配置，不需要修改算法代码。

## Gaussian Splatting后端配置

后端使用官方`graphdeco-inria/gaussian-splatting` revision `54c035f7834b564019656c3e3fcc3646292f727d`，配合本库提供的[主后端补丁](patches/gaussian-splatting.patch)和[CUDA rasterizer补丁](patches/diff-gaussian-rasterization.patch)。补丁包含完整PLY初始化、masked损失与mask读取、配套渲染接口，以及图片demo使用的道路mask合并函数。

**完整克隆、固定版本、应用补丁、GS环境与CUDA编译、自检步骤见[Gaussian后端安装手册](patches/GAUSSIAN_BACKEND.md)。** 基础版本及文件SHA256见[后端清单](patches/gaussian-backend.json)。配置仍使用`gs_backend`和`python_gs`，不需要复制旧实验目录。

配置完成后，在GS环境运行`python -m GLRMap_GS.tools.check_gs_backend --cuda`。它检查版本／补丁／源码，并执行真实的PLY初始化、mask梯度、CUDA渲染反传和优化更新；不能只替换Python文件而继续使用旧CUDA扩展。

## ReSplat依赖与配置

`resplat`是**外部ReSplat SDK的源码根目录**；`python_resplat`是**运行相应阶段的Python可执行文件**，两者不是同一个路径。本仓库不附带外部SDK或预训练权重。

| 使用方式 | ReSplat SDK／权重 | Python环境 |
| --- | --- | --- |
| 查看下载好的视频、PNG和参考模型 | 不需要 | 查看视频／图片不需要训练环境；模型需相应GS查看器 |
| 从100张图片完整重跑demo | **必须安装SDK、应用兼容补丁并下载16视角权重** | `python_resplat`运行新的ReSplat推理／初始化／渲染；`python_gs`训练 |
| 对新窗口运行ReSplat，或使用完整`pipeline.window` | 需要以下SDK、兼容补丁和16视角权重 | ReSplat环境需安装完整SDK依赖 |

### 1. 完整流程使用的SDK与补丁

固定使用[cvg/resplat](https://github.com/cvg/resplat) revision `cae7ddc4cdbd80e05e9f5fa00f5ea02c4e9056b1`。在**全新SDK检出**上执行：

```bash
git clone https://github.com/cvg/resplat.git /path/to/resplat
git -C /path/to/resplat checkout cae7ddc4cdbd80e05e9f5fa00f5ea02c4e9056b1
git -C /path/to/resplat apply --check /path/to/GLRMap_GS/patches/resplat-infer-colmap.patch
git -C /path/to/resplat apply /path/to/GLRMap_GS/patches/resplat-infer-colmap.patch
```

[随库补丁](patches/resplat-infer-colmap.patch)对应本机实际使用的一行修改：精修时向`forward_update`传入`target=None`，目标视图留到后续分块渲染，以减少峰值显存。本机已使用该补丁的SDK不必重复应用；不要直接把其它revision当作此版本替代。

在独立ReSplat环境中，先按[该revision的官方安装说明](https://github.com/cvg/resplat/tree/cae7ddc4cdbd80e05e9f5fa00f5ea02c4e9056b1#installation)配置Torch／CUDA，再安装SDK依赖、固定gsplat和pointops：

```bash
cd /path/to/resplat
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation 'git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3'
(cd src/model/encoder/pointops && python setup.py install)

# 在ReSplat环境及GS环境中分别注册GLRMap_GS。
/path/to/resplat-env/bin/python -m pip install --no-deps --no-build-isolation -e /path/to/GLRMap_GS
/path/to/gs-env/bin/python -m pip install --no-deps --no-build-isolation -e /path/to/GLRMap_GS
```

上游安装示例使用Torch2.7.0／CUDA12.8；本项目实际验证环境为Python3.12、Torch2.7.1+cu118、torchvision0.22.1、gsplat1.5.3、Hydra1.3.2。不要混装两个Torch／CUDA组合；CUDA扩展须在所选环境中编译。GS后端另有自己的训练环境，见下方迁移表。

### 2. 下载16视角权重

本入口使用`dl3dv_16v_540x960` preset，每个窗口均匀选择16个上下文，精修2轮。权重见[上游Model Zoo](https://github.com/cvg/resplat/blob/cae7ddc4cdbd80e05e9f5fa00f5ea02c4e9056b1/MODEL_ZOO.md)，文件约855MiB；它不包含在demo网盘包中。

```bash
cd /path/to/resplat
mkdir -p pretrained
curl -fL 'https://huggingface.co/haofeixu/resplat/resolve/main/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth' \
  -o pretrained/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth
printf '%s  %s\n' \
  'a72dc6d0d56f7ccc50d145c17c83d866d4aaa2c13adb1a22f7a84481001e1a4c' \
  'pretrained/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth' | sha256sum -c -
```

### 3. 填写路径并自检

完整配置中与ReSplat直接相关的字段为：

```json
{
  "resplat": "/path/to/resplat",
  "resplat_checkpoint": "/path/to/resplat/pretrained/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth",
  "python_resplat": "/path/to/resplat-env/bin/python"
}
```

这些字段并入项目根目录的`paths.local.json`，其余字段仍按`paths.example.json`配置。省略`resplat_checkpoint`时默认读取`<resplat>/pretrained/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth`；权重放在其它目录时显式指定它。

在配置好的ReSplat环境中执行以下检查；仅检查路径和导入，不加载权重或启动推理：

```bash
cd /path/to/GLRMap_GS
python - <<'PY'
import sys
from pathlib import Path
from GLRMap_GS.runtime import RESPLAT, python_for
from GLRMap_GS.tools.resplat_runtime import CHECKPOINT
for path in [RESPLAT/'scripts/infer_colmap.py', RESPLAT/'src/model/ply_export.py',
             RESPLAT/'config/main.yaml', CHECKPOINT, Path(python_for('resplat'))]:
    assert path.is_file(), path
sys.path.insert(0, str(RESPLAT/'scripts'))
import infer_colmap
print('ReSplat SDK:', RESPLAT)
print('Checkpoint:', CHECKPOINT)
print('Import check passed:', infer_colmap.__file__)
PY
```

实际入口为`tools/run_resplat_forward_windows.py`，它调用SDK的`scripts/infer_colmap.py`和`src/model/ply_export.py`，并核验相机、原生导出及视频。新ReSplat推理会等待GPU锁及至少18000MiB空闲显存；从图片运行demo同样需要满足此门槛。

## 历史预准备批次入口

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
| [data_preparation/](data_preparation/README.md) | 图片demo快照及SAM推理由`pipeline.demo`负责；另列历史批次资产约定 |
| [sfm/](sfm/README.md) | GLOMAP、固定重叠锚点及相机；900×500图像／K／SIFT同步适配 |
| [matching/](matching/README.md) | 共视选对、RoMaV2双向查询、SIFT／SuperPoint＋LightGlue及身份Snap |
| [triangulation/](triangulation/README.md) | 按共享参考源点构建Track、固定Pose/K求XYZ、几何验收和分块合并 |
| [road/](road/README.md) | 道路Track提取、受支持曲面拟合、射线增密、完整非道路保留；不含专项车道线修复入口 |
| [depth_prior/](depth_prior/README.md) | 可选DA3／Pi3X相机条件化推理、深度约束、替代、Anchor及补点 |
| [gs/](gs/README.md) | 道路GS适配、原生ReSplat坐标／协方差／SH转换、初始化及道路几何固定训练 |
| [visualization/](visualization/README.md) | 单文件离线WebGL、完整点云投影、BEV、GS同轨迹／横移／变道和视频过渡 |
| [pipeline/](pipeline/README.md) | 1000帧点云队列；单窗GS与100帧／25帧重叠队列；环境、进度、恢复与产物交接 |

默认1000帧点云批次重叠100帧；三角化按100张参考图分块但保留全批邻居；GS窗口100帧、步长75、重叠25帧。不能把这三种单位混用。各模块通过包内导入运行，不通过试验脚本修改全局路径来复用算法。

## 迁移所需的项目与配置

外部项目保留独立目录，不自动复制进本仓库。各自的依赖与许可沿用来源项目。

| 配置项 | 依赖／约定 |
| --- | --- |
| `gs_backend` | 按[后端安装手册](patches/GAUSSIAN_BACKEND.md)固定官方版本、应用两个补丁并编译CUDA扩展后的源码根目录 |
| `colmap`, `glomap` | COLMAP 3.9.1 CUDA与GLOMAP；已测GLOMAP revision `99806d0869f802fad218516a2e027793e7ca687d` |
| `romav2_source` | RoMaV2源码；已测 `Parskatt/RoMaV2` revision `95c9968145c8906b7b59383258e9f73b02853d89` |
| `dinov3_source` | DINOv3本地hub源码；已测 `adc254450203739c8149213a7a69d8d905b4fcfa` |
| `roma_assets` | 包含 `models/romav2.0.1.pt` 的资产根目录；路径／SHA256见本库 `config/romav2.json`，权重必须在该资产根下。DINO参数已包含在此checkpoint |
| `sparse_weights` | 含 `superpoint_v1.pth`、`superpoint_lightglue_v0-1_arxiv.pth` 的目录；LightGlue代码需安装，已测revision `c2f8561576f0160b2cdaa3e4634b6b61d6cf6cb8` |
| `resplat` | 独立ReSplat SDK根目录，含src/scripts/config；固定revision与随库补丁见[ReSplat依赖与配置](#resplat依赖与配置) |
| `resplat_checkpoint` | 16视角权重路径；默认`<resplat>/pretrained/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth`，下载与完整SHA256见上文 |
| `gpu_lock` | 同GPU的任务共用同一锁路径；ReSplat推理等待至少18000MiB空闲显存 |
| `python_sam/sfm/matching/resplat/gs` | 五个环境的Python绝对路径；图片demo还需要SAM3.1推理环境 |
| `sam3_source`, `sam3_checkpoint` | SAM3.1源码及multiplex权重；新推理道路、标线、车辆和行人 |
| `pi3x_source`, `da3_source`, `da3_model` | 可选支线的SDK／DA3权重目录；Pi3X权重用推理命令的 `--ckpt` 指定 |

已验证的环境版本（不是完整锁文件）：

| 环境 | 关键版本 |
| --- | --- |
| SfM | Python3.10、Torch2.5.1+cu121、pycolmap4.2.0、NumPy1.26.4 |
| matching | Python3.10、Torch2.7.1+cu118、torchvision0.22.1、pycolmap4.1.0、NumPy1.26.4、SciPy1.15.3 |
| ReSplat | Python3.12、Torch2.7.1+cu118、gsplat1.5.3、Hydra1.3.2、plyfile1.1 |
| GS | Python3.9、Torch2.6.0+cu118、torchvision0.21.0、NumPy2.0.2、plyfile1.1.3；本地编译的rasterization和simple-knn |

换机器还需保存各环境的conda导出／pip freeze和外部项目本地修改；不要直接复制venv或不同Torch版本的CUDA `.so`。原机验证GPU为RTX4090。

当前图片demo只需携带100张图片；mask、相机、点云与ReSplat输出全部由入口新生成。历史`pipeline.window`批处理入口另需原视频、图像及哈希／时间manifest、各mask、首批相机锚点、matching_settings。只从GS继续时还需要道路曲面、来源像素、ReSplat原生PLY和相机等阶段输入。包含软链接的权重、数据及源码须同时迁移链接目标。历史实验的路径／哈希记录保持原样，不改写它们来绕过恢复检查。

## 检查

在匹配环境（含pycolmap）中执行：

```bash
python -m GLRMap_GS --help
OPENBLAS_NUM_THREADS=1 python -m unittest \
  GLRMap_GS.triangulation.test_contracts GLRMap_GS.road.test_contracts \
  GLRMap_GS.depth_prior.test_contracts GLRMap_GS.gs.test_contracts \
  GLRMap_GS.visualization.test_contracts GLRMap_GS.pipeline.test_contracts -q
```

2026-10-26迁出后88项CPU检查通过，独立入口重跑394–396秒完整流程耗时133秒，400步GS及渲染均完成（显式复用已审计SfM，没有重跑1000帧SfM）。本机记录见 `experiments/sanhuan_394_396_migration_check/progress.json`。394–396秒20视图完整流程曾完成46对RoMa、231405个三角化点、125543个拟合道路点、400步训练及全部渲染；执行通过不代表其它场景画质自动达标，应查看PNG及左右外插视频。

`CORE_PIPELINE.md`与各功能README保留算法契约。迁移前的 `repository/`、`external_sources/`、`reference_configs/`、来源清单仅作为本地历史快照保留，已从Git交付范围排除；生产模块不导入它们。模型、实验、数据和本机配置也不进入Git。
