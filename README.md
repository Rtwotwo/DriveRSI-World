1.严谨跨clones进行跨文件夹的导入，必须集成为我们的自身的代码设计，

2.现有的代码必须需要按照分门别类有层次化的放置，保证我们的代码保存干净整洁；

3.现有代码里面存在大量的backend的子文件夹，我不清楚是干什么的！但是我需要你能够清理掉的就清理掉；

4.现有就使用Alpamayo模型作为教师模型，AUGUstus这个模型暂时作为学生模型的，用于后面的RSI核心的难例的学习！！！

5.中间任何的输入输出产生的缓存文件仅仅允许放在exp里面，不得随便放置

6.不要随便增加md文件或者其他的日志文件，不能随便新增.XXX文件夹！！

# DriveRSI-World

难例场景与观测合成平台。当前技术底座是已纳入自身命名空间的 **WorldEngine / MTGS**；与 **DriveRSI-Policy** 分别维护在 `main`，没有模型权重或策略训练代码依赖。

## 代码分层

| 位置 | 职责 |
| --- | --- |
| `src/data/`、`src/contracts/` | 资产、来源、道路/对象契约、官方训练/评测隔离、完整时间窗 |
| `src/simulation/` | 自有 WorldEngine 场景、交通、MTGS 渲染、原生 PDM 评分与短时执行 |
| `src/synthesis/` | 实际轨迹准入、来源条件下的合成多样性与速度规则 |
| `src/quality/` | PSNR、动态对象同分支绑定、数值与视觉证据验收 |
| `src/runtime/`、`src/jobs.py` | exp 路径约束、外部策略 IPC、阶段执行和哈希回执 |
| `src/resources/navsim/` | 与学生模型解耦的官方划分表和来源/许可证 |

前馈、4DGS 或其他合成器是后续可替换的实现，当前没有假装实现这些方法。新实现应输出相同的来源、观测、执行轨迹和质量证据，不能因渲染更快而跳过验收。

## 使用

```bash
export PYTHONDONTWRITEBYTECODE=1
mkdir -p exp/build exp/install/cache exp/install/tmp
export PIP_CACHE_DIR="$PWD/exp/install/cache" TMPDIR="$PWD/exp/install/tmp"
python -B -m pip install --no-deps --no-build-isolation -e .
python -B -m driversi_world inspect --output "$PWD/exp/inspect"
python -B -m driversi_world mine --sources "$PWD/exp/input/sources.json" --scores "$PWD/exp/input/train-scores.csv" --limit 2 --output "$PWD/exp/selection"
```

`inspect` 只核验独立入口，不是 CUDA 证书。原生仿真另需兼容的 PyTorch、gsplat、Hydra、OpenCV、nuPlan、地图依赖。nuPlan 作为普通安装包使用（当前本机为 1.2.0 wheel），不能把 Policy 内的 nuPlan 源目录加入 PYTHONPATH。官方划分来自固定版本 NAVSIM，不能通过更换路径将评测数据用于训练。

`sources.json` 使用 `driversi.worldengine.selection.v1`，含 `split=navtrain`、`query_split=navtrain`、`records=[{token,log_name,source_path}]`。只读取操作员可信的原始 pickle 日志。失败按原生分数和碰撞/道路/TTC 子项选择，并按采集记录去重；不是语义检索 Top-K。

外部调度通过 `python -B -m driversi_world convert|rollout|export --config <任务JSON> --output <本仓库exp子目录>` 调用。阶段产出 `driversi.world.receipt.v1`，记录文件哈希、实际状态、耗时和依赖；Policy 用版本化接口消费。World 不直接导入 Policy 源码；`policy_command` 是安装后的策略 worker argv，通过 socket 返回动作，可替换其他策略。

## 验收边界

只选择训练来源；固定评测及同源记录排除。转换后检查四帧历史＋八帧未来的完整 2 Hz 窗口。上游当前未完整评分行人/骑行者，因此含这些对象的转换场景继续拒绝，不计为安全通过。当前执行是短时窗口和理想轨迹跟踪，不代表长路线驾驶成功率。

`driversi.worldengine.candidates.v1` 始终 `trainable=false`。PDMS 严格大于 0.95 只是分数初筛，仍需源视角 PSNR 大于 25 dB、动态对象与轨迹同分支、教师同起点实际执行、训练标签一致性等证据。教师生成与参数晋级属于 Policy；本项目不会把高分候选直接升级成训练许可。

所有新结果、缓存、临时文件、日志和图像仅在 `exp/<实验ID>/`。不复制共享数据，不自动下载全库，不跨 clones 导入。WorldEngine Apache-2.0 许可证及原固定版本来源保留在 `src/simulation/`；本次提取来源记录在 `provenance.json`。
