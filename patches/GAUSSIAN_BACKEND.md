# Gaussian Splatting后端：版本、补丁与安装

GLRMap_GS使用修改过的Graphdeco后端。以下步骤从官方源码构建本库所需的后端，不需要复制旧电脑上的整个`gaussian-splatting`工作区。现有训练模型、实验数据和二进制扩展不放进补丁。

## 1. 固定版本

| 源码 | 基础revision | 本库补丁 |
| --- | --- | --- |
| [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) | `54c035f7834b564019656c3e3fcc3646292f727d` | [gaussian-splatting.patch](gaussian-splatting.patch) |
| `submodules/diff-gaussian-rasterization` | `9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0` | [diff-gaussian-rasterization.patch](diff-gaussian-rasterization.patch) |
| `submodules/simple-knn` | `86710c2d4b46680c02301765dd79e465819c8f19` | 无修改，重新编译 |
| rasterizer内的`third_party/glm` | `5c46b9c07008ae65cb81ab79cd677ecc1934b903` | 无修改，头文件依赖 |

[gaussian-backend.json](gaussian-backend.json)记录固定版本、两个补丁的SHA256和15个修改／新增文件的SHA256。保留上游源码及其许可证。

## 2. 获取源码、应用两个补丁

以下路径替换为本机位置。`/path/to/gaussian-splatting`应为全新检出目录；当前已经使用修改版后端的电脑不必重复应用补丁。

```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting.git /path/to/gaussian-splatting
git -C /path/to/gaussian-splatting checkout --detach 54c035f7834b564019656c3e3fcc3646292f727d

# 当前GLRMap_GS训练只需这两个CUDA子模块及rasterizer内部的GLM。
git -C /path/to/gaussian-splatting submodule update --init --recursive -- \
  submodules/diff-gaussian-rasterization submodules/simple-knn

git -C /path/to/gaussian-splatting apply --check /path/to/GLRMap_GS/patches/gaussian-splatting.patch
git -C /path/to/gaussian-splatting apply /path/to/GLRMap_GS/patches/gaussian-splatting.patch

git -C /path/to/gaussian-splatting/submodules/diff-gaussian-rasterization \
  apply --check /path/to/GLRMap_GS/patches/diff-gaussian-rasterization.patch
git -C /path/to/gaussian-splatting/submodules/diff-gaussian-rasterization \
  apply /path/to/GLRMap_GS/patches/diff-gaussian-rasterization.patch
```

两个补丁配套使用：Python渲染器期待扩展后的返回值，旧的CUDA扩展不能直接配套。

## 3. GS环境与CUDA编译

本机验证组合为Linux、Python3.9.25、Torch2.6.0+cu118、torchvision0.21.0+cu118、CUDA Toolkit11.8、GCC9.4.0、RTX4090。GS训练环境与ReSplat环境独立。

先安装Git、GCC/G++及包含`nvcc`的CUDA Toolkit11.8；PyTorch自带的CUDA运行库不能代替编译工具。不要沿用上游旧`environment.yml`中的Python3.7／Torch1.12配置。

```bash
conda create -n glrmap_gs_backend python=3.9 -y
conda activate glrmap_gs_backend

python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r /path/to/GLRMap_GS/config/gaussian-backend-requirements.txt

# 按实际Toolkit安装位置填写；nvcc版本应与上述cu118对应。
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'

# 8.9对应本机RTX4090；其它GPU使用自身计算能力，或省略此变量让Torch检测。
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=2

python -m pip install --no-deps --no-build-isolation \
  /path/to/gaussian-splatting/submodules/diff-gaussian-rasterization \
  /path/to/gaussian-splatting/submodules/simple-knn
python -m pip install --no-deps --no-build-isolation -e /path/to/GLRMap_GS
```

请在应用补丁之后编译。已安装旧版扩展而需要重装时，在上述扩展安装命令中加`--force-reinstall --no-cache-dir`。以检查脚本输出的实际模块路径确认当前Python加载了哪个扩展。

`fused-ssim`和SIBR不是当前GLRMap_GS训练入口的必需项；这里使用补丁中的masked SSIM。完整训练循环由`GLRMap_GS.gs.training`执行。

## 4. 配置与自检

在`paths.local.json`中填写：

```json
{
  "gs_backend": "/path/to/gaussian-splatting",
  "python_gs": "/path/to/conda/envs/glrmap_gs_backend/bin/python"
}
```

这两项并入原有配置，其余SAM、SfM、RoMa、ReSplat路径保持完整。图片demo的道路mask合并由SAM环境调用后端中新增的`complete_sanhuan_sam31_road_masks.py`；该文件已包含在主补丁中。

在GS环境执行：

```bash
# 检查Git基础版本、源码文件和补丁校验值，不使用GPU。
python -m GLRMap_GS.tools.check_gs_backend

# 额外执行小型真实CUDA验收，临时输入和输出会自动清理。
python -m GLRMap_GS.tools.check_gs_backend --cuda
```

也可传`--backend /other/backend`检查指定目录，传`--output report.json`保存报告。CUDA检查包含：完整PLY初始化、曝光／训练状态、mask无效像素零梯度、有效mask加载、扩展渲染返回值、渲染反传及一次优化更新。

## 5. 补丁具体内容

| 文件 | 当前流程需要的内容 |
| --- | --- |
| `scene/__init__.py` | `Scene(..., init_ply=...)`，从完整高斯PLY初始化 |
| `scene/gaussian_model.py` | `load_ply_for_training`初始化曝光、半径缓存及学习率尺度；保留当前的剪枝临时状态检查 |
| `utils/loss_utils.py` | 有效像素归一化的masked L1、阻断无效预测像素梯度的masked SSIM |
| `utils/sky_mask_utils.py` | `ValidMaskSupervisor`读取、匹配及缩放有效mask；文件名为历史命名，当前demo只加载动态有效mask |
| `gaussian_renderer/__init__.py`及rasterizer补丁 | 配套返回逆深度、mode深度／ID／权重与累计alpha；CUDA和Python接口同步 |
| `arguments/__init__.py` | 保留当前学习率日程参数默认值；GLRMap_GS训练明确传入阶段长度 |
| `complete_sanhuan_sam31_road_masks.py` | 已有道路／附近标线mask合并函数，供图片demo调用 |

道路参数冻结策略、两阶段训练、道路逆深度监督、训练预算和checkpoint恢复在GLRMap_GS内实现。当前图片demo设置`road_train_appearance=true`：仅固定道路位置、尺寸、旋转，训练SH颜色与透明度；旧配置未设置此项时仍冻结道路全部参数。旧工作区的`train.py`、`render.py`及其它实验专用修改不作为本库运行依赖打包；直接使用官方`train.py`不会自动执行本库的训练策略。

实际验证记录见[gaussian-backend-validation.json](gaussian-backend-validation.json)。其中短程训练使用现有100帧输入做后端集成验收，不作为新的画质参考结果。
