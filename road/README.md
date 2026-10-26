# 道路处理

本目录已整理当前三环的**原始道路Track提取 → 局部曲面拟合 → 像素射线增密 → 完整点云合成**。算法以当前`dense_road_v2`为准，保持原有拟合、支持域、采样和选点规则；原实验、相机及源码快照保持不变。

## 代码职责

| 文件 | 负责什么 |
| --- | --- |
| [selection.py](selection.py) | `extract_road`从三角化点及其保留观测提取道路，保留完整点云中的来源行 |
| [surface.py](surface.py) | 相机道路局部基、五系数曲面拟合／求值／法线及稳定射线求交 |
| [fitting.py](fitting.py) | `fit_patches`拟合与留出验收；`save_patch/load_patches`保存和读取明确的时间支持 |
| [sampling.py](sampling.py) | `query_surface`统一曲面深度查询；`sample_road`采样取色；`select_cells`按空间单元保留真实交点 |
| [reconstruct.py](reconstruct.py) | `run`串联以上阶段，导出稠密道路、完整合成点云及报告 |
| [test_contracts.py](test_contracts.py) | 道路观测、拟合验收、射线支持、来源与背景保护检查 |
| [lane_protection.py](lane_protection.py) | 真实标线Track投票及局部拟合补点避让，不移动真实匹配点 |

新代码使用已整理的`sfm.geometry`、`matching.sparse.valid_mask`及`io_utils`，不再通过旧实验入口导入曲面数学。旧的“只把已有道路点投到曲面上”的流程没有作为第二条生产路径带入。当前数学模型仅为五系数曲面，没有加入历史样条／七系数格式。

## 三角化输出如何交接

输入是包含`manifest.json`的三角化实验目录。默认读取`B1_shared/new_points.npz`、`point_observations.npz`和`metrics.json`，图像、相机及mask位置沿用manifest的`source`。

- 使用`sfm.geometry.model_data`相同的图像名排序；观测`image_index`是此排序下标。
- 一个原始点至少有2条保留观测落入静态道路，且道路观测比例≥75%，才进入`ROAD_shared`。道路mask采用原有`>128`，静态有效mask采用`>0`。
- `ROAD_shared`保留原XYZ、RGB、Track长度、夹角与误差字段；`road_point_selection.npz`保留`source_point_index`和道路观测数量。
- 拟合所用参考帧来自该点第一条保留观测；共享源点Track保留参考源观测且位于首位，因此与上游参考源身份一致。

`extract_road(root, destination, branch='B1_shared')`将结果写到新位置，返回道路点、原完整点云行号和参考帧数组，直接交给拟合阶段。输入目录保持只读。

## 曲面与验收规则

1. 用至少4视图、夹角≥3°、平均重投影≤0.5px、最大重投影≤1px的道路Track作为拟合证据。
2. 每100帧建立一块，范围为`[core-20, core+120)`，在序列边界截断；中间块覆盖14秒，相邻块重叠40帧。该单位与100帧GS窗口、25帧GS重叠不同。
3. 从相机轨迹建立前向、横向、向上的局部基，拟合归一化坐标下的`h=c0+c1*u+c2*v+c3*u²+c4*u*v`。允许纵向弯曲和变化的横坡；不是全局平面，也不是完整地形模型。
4. 保留原RANSAC初始化、密度平衡和12轮鲁棒迭代；随机种子为`54+core`。
5. 按完整点云来源行`source_point_index % 5 == 0`确定约20%留出点；留出点不参与拟合。训练误差≤`3*GRID`的一致点必须≥1000，再用它们的0.25–99.75%分位确定支持范围。
6. 在该支持范围加0.05 SfM边界内，受支持留出点必须≥500，法线方向误差P90≤`5*GRID`。未通过就失败，不跳过失败块继续声称完成。

`GRID=0.00420966`保持现有值，单位为SfM单位；训练门槛约0.012629，留出P90门槛约0.0210483，不能直接称为米。该留出仅检验曲面对已有RoMa道路的拟合一致性，不是独立三维真值。

每份`surfaces/patch_XX.npz`包含曲面参数、`origin/basis`，并新增明确的`first/stop`。读取时用实际范围，不按文件序号猜测。`load_patches`要求这些字段；旧产物继续保留原位，未自动改写或添加兼容推断。

## 射线、颜色与空间选点

`query_surface(patches, view_index, xy, pose, intrinsics)`统一以下规则：

- 以原相机K和world-to-camera Pose生成射线。返回深度是**相机Z**，不是单位射线距离；支持SfM的3×4和GS的4×4 Pose。
- 查询当前帧所在的曲面块，选择正的近处解；平面退化使用精确线性根。只接受空间支持内的有限正深度，射线与法线的绝对余弦≥0.03。
- 重叠曲面沿用时间权重`min(1,(i-first+1)/20,(stop-i)/20)`融合相机Z和入射余弦。融合后的点仍在同一条源射线上。
- 返回整批`xyz/depth/cosine/valid`；无支持的位置为无效且值为NaN，调用方按valid构造深度监督／覆盖mask。

该函数可用于900×500增密和666×370 GS道路深度。两处原来重复的支持判断和加权规则已经集中到此处，正在运行的GS入口尚未切换。

道路增密使用静态道路像素每2像素采样，RGB直接取同一张图的同一像素。**保持原增密约定：整数`(xs,ys)`直接作为射线坐标，不补0.5。** 上游匹配的SP／分层像素中心仍沿用其自身0.5约定，不能在此次整理中统一偏移。

以首相机中心为网格原点，单元边长`GRID/2=0.00210483`；每个单元保留`camera_Z/sqrt(cosine)`最小的真实候选，兼顾较近和较正的视角。选点不把XYZ移动到体素中心。输出保留`source_image_index`与`source_xy`，可回查原图颜色及源射线。

增密点来自曲面求交，**没有新增匹配Track**，不能继承原Track的重投影精度或质量等级。`FUSED_AFTER`由原始非道路XYZ/RGB和增密道路组成，原始道路另存于`ROAD_shared`；完整原始三角化仍在输入目录。

### 标线重点匹配后的保护试验

`lane_protection.marking_votes`在真实Track的保留二维观测采样标线候选mask（1px容差），至少两票且占保留观测一半才标记；仍须先通过正常三角化和道路分类。`clear_surface`仅移除源像素落入3px标线带的拟合点，以及距离真实标线道路UV投影≤`2*GRID`的拟合点；后者忽略拟合高度，防止错误高度层覆盖标线。必须传入同一个局部道路窗口的`origin/basis`，不能把这一UV判定直接用于全局交叉道路或立交。

330–340秒对照的`finish_experiment.py`显式调用以上两步，最终为**完整真实匹配点＋保护带之外的拟合补点**，原XYZ/RGB逐项保留；`point_origin=0`表示真实Track，`1`表示无Track的曲面补点。仅仅增加拟合权重不能代替这条保护规则。本次沿用原拟合曲面以隔离变量，没有重新拟合、运行PDA或训练GS；默认`road.reconstruct.run`仍保留上文已整理的旧行为，尚未批量替换其它已完成实验。

## 调用与产物

在仓库根目录调用，输出必须为一个尚不存在的目录：

```bash
python -m GLRMap_GS.pipeline.reconstruct road /path/to/triangulation_experiment --output /path/to/new_road_output
```

Python入口为`road.reconstruct.run(root, output, branch='B1_shared')`。已有`triangulation.batch`输出可以直接作为输入，不要求先运行旧道路脚本。道路分割mask须已由数据准备阶段提供，本轮不重新运行SAM。

```text
new_road_output/
  manifest.json                    进行中／失败／完成状态及输入位置
  ROAD_shared/                     原始道路Track点和完整点云来源行
  surfaces/patch_XX.npz             含明确窗口范围的已验收曲面
  fit_report.json                  各块训练／留出统计
  ROAD_fitted/new_points.npz/.ply   增密XYZ/RGB、源图像与像素
  FUSED_BEFORE/                    指向只读原始完整点云
  FUSED_AFTER/new_points.npz/.ply    原始非道路＋增密道路
  dense_report.json                逐帧交点、空间采样及输入哈希
```

本入口只负责几何与统计；图表／HTML已整理到`visualization/`，由`pipeline.cloud`的report阶段调用。ReSplat道路高斯适配已在`gs/`，作为后续独立阶段。车道线专项修复仍在快照中，未接入此路径。使用顺序与输出交接见[Pipeline手册](../pipeline/README.md)。

## 实际核对

```bash
OPENBLAS_NUM_THREADS=1 python -m unittest GLRMap_GS.road.test_contracts -v
```

9项道路检查通过。另以现有`sanhuan_340s610s_glomap1000_roma_20261022/block_00`做只读数据核对，计算产物只写临时目录：

- 对`roma/chunk00_shared`重新分类，72313个道路点与整理前的选择完全一致；NPZ字段和PLY字节相同。
- 用原始道路Track分别执行整理前后的全部10块拟合，所有曲面参数、局部基及支持域逐项完全一致，且与现有保存曲面一致；训练／留出统计相同。
- 在`003200/003290/003300/003700/004195`五张900×500相机上检查89719个有效交点，valid、相机Z、float32 XYZ及选点评分完全一致；合并这些候选后的46991个空间选中行也一致。
- 在`003200/003290/003700`三张666×370相机上，使用原GS深度循环核对相同静态道路像素，共用查询的valid和深度完全一致。
- 第一张图在900×500采样中只有21个受支持交点，在666×370原生道路查询中为48个。这是原曲面支持边界的既有行为，本轮没有扩展支持来填补边缘。
- 1832份原快照SHA256保持不变。未重跑完整1000帧增密导出，没有启动GLOMAP、GPU匹配或GS训练。

### 2026-10-25：394–396秒实际复现补充

历史复现记录：原工程 `superDIM/experiments/sanhuan_394s396s_review_replay_20261025/README.md`（不随本库发布）。在原相机／原始道路Track／mask不变的条件下，用本模块重新拟合全部10块并执行完整1000帧道路增密：24784234个有效射线交点、6064434个选中点，XYZ/RGB/source_image_index/source_xy与旧`dense_road_v2`逐项完全一致。394–396秒BEV也一致，已有边缘缺陷没有修复。

同一实验另外重建20张参考图的Track及三角化，226813个点的几何／颜色／误差字段与旧结果完全一致。两项分别复用对应上游缓存，未运行新的SfM／GPU匹配／GS，也未验证共视计数修复后新配对的完整流程。
