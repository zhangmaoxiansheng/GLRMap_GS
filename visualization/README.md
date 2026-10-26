# 可视化与渲染

已提取点云三维HTML、同相机完整投影、同尺度BEV／侧视，以及GS原轨迹、定点横移、前进变道和重叠视频预览。所有展示从明确的点云／相机／视频输入生成；没有旧DIM对照路径、试验目录常量或旧队列导入。

完整运行顺序与输入目录交接见[Pipeline手册](../pipeline/README.md)。本页负责展示数据契约与输出，不是训练入口。

## 模块归属

| 文件 | 负责什么 |
| --- | --- |
| [clouds.py](clouds.py)、[pointcloud_viewer.html](pointcloud_viewer.html) | 原生离线WebGL、16字节XYZ/RGB/quality数据、相机导航、并排比较与点数标注 |
| [projection.py](projection.py) | 原XYZ/RGB针孔投影、同像素最近正深度、像素／网格覆盖统计 |
| [bev.py](bev.py) | 统一道路相机基底、相同比例／画幅／点大小的BEV和侧视，不压平道路 |
| [report.py](report.py) | 点云展示入口，组合上述模块；保存PNG、统计JSON及可独立打开的HTML |
| [trajectory.py](trajectory.py) | 当前相机的定点横移与前进变道；固定旋转/K，位移单位为SfM单位 |
| [gs.py](gs.py) | 原图／ReSplat／初始化／训练后对照、原轨迹、三个定点横移及前进变道渲染 |
| [video.py](video.py) | FFmpeg编码、解码／PTS核验、按相同全局帧号溶解重叠视频 |
| [html.py](html.py) | 图片／视频data URI及完整离线报告；拒绝index页 |

GS审计继续使用`gs/rendering.py`这一份gsplat参数，不重复实现坐标转换或渲染器。ReSplat模型推理、训练、点云筛选、DA3/Pi3X可靠性判定仍属于对应算法模块，不放入展示层。

## 点云展示约定

- 默认从实验`manifest.json`的`viewer_branches/viewer_labels`读取明确分组；也可用`--branches`指定。没有EXP54、Loop或DIM默认实验。
- 三维展示默认每组最多250000点，沿用固定种子`20261015`、排序后的无放回抽样；`portable_max_points`可逐组指定。只采样展示缓冲，HTML同时标注展示／原始数量，不回写点云。曲面／前馈补点的`has_track_observations=false`保留，不伪装成匹配Track。
- **图像投影使用完整点云**，所有组共用同一Pose、K、画幅和1像素点，正深度／画幅内筛选并取同像素最近点。完整投影不按道路、静态或天空mask裁剪；道路ROI仅用于补充覆盖统计。道路与多视图道路可作为显式分组传入，不能替代完整分组。
- BEV和侧视使用全量输入，同一相机基底和坐标范围。显示轴界限采用合并各组0.2/99.8百分位并加边距以便观察；没有修改XYZ、压平或重新拟合道路，统计保留全量输入数量。
- PNG、JSON和HTML保存在指定输出；原图／mask、相机、点云／指标的输入哈希在前后核对。已有同名三维HTML会另存历史副本。HTML内嵌点云或图片／视频，不依赖相邻文件、CDN或本地服务；不生成导航index。
- 投影密度、训练视图L1和与先验一致性不是独立三维精度；稀疏点云z-buffer不是完整遮挡几何，横移视角没有GT。

```bash
# 建议对整理后的road输出生成展示；不会调用匹配、拟合或训练。
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/sanhuan_plot_cache \
  python -m GLRMap_GS.visualization.report ROAD_OUTPUT \
  --branches FUSED_BEFORE FUSED_AFTER ROAD_shared ROAD_fitted \
  --image-name 003700.jpg --max-points 250000 --output NEW_COMPARISON
```

产出`raw_3d_interactive.html`、`camera_projection.html`、`bev_projection.html`；三份分别是完整三维、完整投影、投影＋BEV内容，均可单独下载。另存`camera_projection.png`、`road_bev_side.png`及对应统计。指定相机图像名必须存在于本场景；省略时选按名字排序的中间帧，不套用旧场景帧号。

## GS渲染与视频

```bash
/home/luvision/anaconda3/envs/resplat/bin/python -m GLRMap_GS.visualization.gs \
  --config NEW_PIPELINE_CONFIG.json
```

要求整理后的GS训练完成且输入／模型哈希通过核验。输出仍在该配置的GS目录：

- `trained_forward.mp4`：原轨迹；`original_resplat_init_trained_forward.mp4`：原图／原生ReSplat／混合初始化／训练后。
- `init_vs_trained_lateral.mp4`：默认锚点30/60/85，各96帧、24FPS；`forward_lane_change.mp4`：100个原相机前进中左右横移，10FPS。相机NPZ、审计和抽查JPG一并保存。
- `render_comparison.html`内嵌四段视频，`visualization_complete.json`记录视频、审计、HTML和相机产物哈希。GPU锁内释放模型及缓存，避免两次渲染之间占用下一任务的显存。

`video.join_forward`直接接收`[{first_frame, stop_frame, video}, ...]`。重叠区默认中间10帧按同一个全局时间溶解，输出逐帧来源与权重；全程无丢帧／重复全局帧，PTS必须为`i/10`。这里只验证视频区间；相机和共同道路行的真实训练重叠由`pipeline/gs_queue.py`验证。视频预览**不融合高斯模型**。

## 核对与限制

新增8项CPU检查覆盖最近深度／RGB、覆盖统计、离线HTML及打包抽样身份、全量投影不受零静态mask影响、BEV全量计数、横移约束和过渡帧连续性。结合其它模块共75项检查通过。

实际缓存只读核对：12000个完整场景点的新旧投影逐项一致；三个96帧定点横移及100帧变道的Pose/K与原缓存完全一致；1000帧预览的全帧来源和12处溶解计划与原记录一致。另在临时目录使用真实固定相机与实际XYZ/RGB子集完成三维／投影／BEV报告，并用两段合成视频核对175帧、10FPS、逐帧PTS及溶解颜色。

WebGL脚本通过Node语法检查，离线payload与资源引用通过检查；本机没有可用的浏览器，未做浏览器截图／交互验收。没有重新渲染真实GS视频或启动CUDA实验；原1832份源码快照未改动。
