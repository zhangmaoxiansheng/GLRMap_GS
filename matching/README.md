# 图像匹配与Snap

负责图像对选择、RoMaV2推理、SIFT／SuperPoint＋LightGlue辅助关联及Snap。输出带图像、像素、置信度和共享观测标识的匹配；Track合并及三维求解归属`triangulation/`。

整理后的实现：

| 文件 | 当前职责 |
| --- | --- |
| [pairs.py](pairs.py) | 当前共视评分、生成树、邻居补足和固定预算选对 |
| [geometry_guided.py](geometry_guided.py) | `match_all`：特征、双向RoMa查询与身份Snap主流程 |
| [romav2_runtime.py](romav2_runtime.py) | 现有离线模型／权重加载与稠密场采样 |
| [seeds.py](seeds.py) | 固定源种子及查询；不再导入三角化实验脚本 |
| [sparse.py](sparse.py) | SIFT身份读取、SP/LG离线权重、mask采样和确定性随机种子 |
| [snap.py](snap.py) | 双端证据检查、源身份保持和精度矩阵单位变换 |
| [filters.py](filters.py) | 匹配、Track和三角化共用的一份筛选参数读取规则 |

历史来源（不作运行入口）：

- [基于稀疏几何的图像选对](../matching/pairs.py)
- [RoMaV2推理](../matching/romav2_runtime.py)
- [几何引导匹配](../matching/geometry_guided.py)
- [Snap核心](../matching/snap.py)

当前生产选对仅用共视策略，不自动启用DA3／voxel。旧DIM匹配分支不纳入整理。缓存字段及阈值见[核心整理说明](../CORE_PIPELINE.md)。

共视数量按每张图的**唯一Point3D ID集合**计算；实际GLOMAP中一图可能有同ID重复观测，不能对含重复值的数组使用`intersect1d(..., assume_unique=True)`。整数稀疏可见性矩阵乘积一次得到全图共同点计数；生成树、补邻居和预算规则沿用既有逻辑。2026-10-24修正后，已核对批次的3600对中有186对变化，旧缓存配对不会被回写；详见[本轮复查](../README.md#2026-10-24完整主链复查)。

运行顺序、参数来源与缓存恢复见[Pipeline手册](../pipeline/README.md)。

车道线加密对照可在manifest中显式设置`prepared_source_seeds=true`，提供每图固定的`shared/seeds/*.npz`（`xy/kind/key/image_id`）。`prepared_seeds`检查预算、图像ID及唯一源身份；`match_all`不重新生成或移动它们，只处理本次图像对实际使用的视图。默认SIFT/SP/分层采样方案不变。330–340秒试验保留原20k前缀，再追加5000个静态有效标线带像素，仍共享同一源ID，详见`experiments/sanhuan_330s340s_lane_priority_roma_20261024/`；这不是默认给所有区域提高点数。
