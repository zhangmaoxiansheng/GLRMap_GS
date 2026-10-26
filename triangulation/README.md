# Track与三角化

负责共享观测ID、跨图像对Track合并、固定SfM相机的三角化，以及夹角、重投影等几何筛选。保留真实观测关联与来源；深度预测补点不能冒充多视图Track。

整理后的实现：

- [tracks.py](tracks.py)：`build_shared`按共享参考源点构造直接匹配星形Track，保留真实候选和观测证据。
- [geometry.py](geometry.py)：批量投影／Jacobian、DLT、XYZ优化、鲁棒初始化及证据约束的异常处理。
- [reconstruct.py](reconstruct.py)：固定相机三角化、严格双视图门槛及带来源的产物导出。
- [batch.py](batch.py)：按参考图分块，保持全邻居连接；合并时按候选Track总数偏移ID。
- [test_contracts.py](test_contracts.py)：相机、查询、Snap、Track及分块合并的针对性检查。

历史来源（不作运行入口）：

- [共享观测、Track及三角化](../triangulation/tracks.py)
- [现有稠密流程中的基础实现](../triangulation/geometry.py)

可选深度约束的预测读取和置信度处理已在`depth_prior/`，通过`pipeline.prior`显式启用；本三角化主链不自动接入。当前批处理关闭留出计算，质量标签为0；保留显式请求留出诊断的底层能力。调用方法及数值等价结果见[核心整理说明](../CORE_PIPELINE.md)。

运行顺序与输入根目录见[Pipeline手册](../pipeline/README.md)。

增量源点试验使用`build_shared(..., minimum_seed_index=20000)`，只构建新增源ID的Track，源ID不重编号。与原始结果合并时复用`merge_chunks`，按候选Track数量偏移新增Track ID；原20k点的全部字段及观测逐项保持一致。默认下限为0，原流程不变；该增量接口不合并不同源星形Track，也不放宽任何几何门槛。
