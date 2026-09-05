# Tank RL Simulator

这是一个坦克游戏模拟与强化学习项目。游戏核心、显示渲染、模型观察和训练代码彼此分开；强化学习层不会向核心加入内置决策行为。

## 目录结构

```text
tank_rl_sim/
├─ core/
│  ├─ game.py       游戏循环、移动、碰撞、开火、胜负
│  ├─ maze.py       随机迷宫与墙壁数据
│  ├─ geometry.py   线段与矩形相交
│  └─ entities.py   坦克和子弹的数据结构
├─ renderer.py      Pygame 显示，只读取 core 的状态
├─ demo.py          键盘试玩入口
├─ rl/               强化学习（迷宫对战）
├─ supervised/       监督学习（模仿寻路开火人机）
│  ├─ train.py       行为克隆
│  ├─ watch.py       看人机或克隆模型
│  ├─ evaluate.py    评估克隆模型
│  └─ teachers.py    寻路开火人机导师
├─ tools/            独立视频测速工具
└─ tests/            核心和强化学习组件测试
```

核心按照 24 Hz 固定时间步运行。每次调用 `game.update(...)`，世界前进一个物理帧。核心不包含任何内置决策行为，所有坦克的控制都必须由外部传入。

当前规则：每辆坦克都是独立个体，不存在队伍；坦克只有一条命，任意子弹命中一次就立即被摧毁。车身实测长 `0.4559` 格、宽 `0.3389` 格，炮管使用固定旋转矩形显示，长度为 `0.26795` 格、宽度为 `0.091` 格，实测最大线速度为 `1.8622` 格/秒。子弹直径为 `0.09` 格、实测速度为 `2.2738` 格/秒、寿命为 10 秒，最大射击间隔为 `0.234` 秒（约 `4.2735` 发/秒），每辆坦克最多同时保留 5 颗子弹。子弹撞墙会反弹，反弹回来也能击中发射者本人。地图内外墙壁统一使用实测的 `0.0735` 格宽度。

坦克贴墙转向时不会被禁止旋转；如果旋转后的矩形与墙重叠，核心会将坦克沿最短方向推出墙面。

控制格式：

```python
control = {
    "throttle": 2,  # 0=后退，1=停止，2=前进
    "steer": 0,     # 0=左转，1=不转，2=右转
    "fire": 1,      # 0=不开火，1=开火
}
```

也可以使用等价元组：`control = (2, 0, 1)`。

推进一帧时，需要按 `game.tanks` 的顺序传入每辆坦克的控制：

```python
game.update([
    (2, 0, 1),  # 0 号坦克
    (1, 1, 0),  # 1 号坦克
])
```

## 运行

```powershell
conda activate teacher
cd D:\bysq\tank_rl_sim
python demo.py
```

`W/S` 控制前进和后退，`A/D` 控制左右转向，`J` 发射；同时保留方向键和空格。`R` 随机重置地图，`Esc` 退出。

运行测试：

```powershell
python -m pytest -q
```

## 模型训练使用方法

当前推荐：先用寻路开火人机做监督预训练，效果差不多后再做强化学习。

开始前进入项目和 Python 环境：

```powershell
conda activate teacher
cd D:\bysq\tank_rl_sim
```

### 1. 固定离线数据与监督预训练

两边都由人机开车，模型只学它的油门、转向、开火。新版 hunter 会 A* 绕墙；直射被挡住时会用镜像目标生成反弹候选，再用真实墙体和子弹尺寸完整预演弹道；躲避时会同时预测所有敌我子弹未来 1.25 秒（包含后续反弹），在 9 种油门/转向组合中选安全余量最大的动作。

先固定采集一次数据。一个地图种子的完整对局只会进入一个集合；下面使用 800 张训练地图和后续 200 张验证地图，并按 20 局一个压缩分片保存：

```powershell
python -m supervised.collect `
  --output datasets/hunter_pixel_v5 `
  --seed-start 10000 `
  --train-seeds 800 `
  --validation-seeds 200 `
  --episodes-per-shard 20 `
  --workers 4
```

清单写在 `manifest.json`。如果输出目录已经存在清单，采集器会拒绝混写，必须显式换目录。分片同时保存地图种子、完整对局边界和双方逐步动作，因此以后更换观察编码时可以按相同轨迹重新编码。

然后只读取这份固定数据训练；每轮完整跑一次互不重叠的验证地图：

```powershell
python -m supervised.train_offline `
  --dataset datasets/hunter_pixel_v5 `
  --epochs 20 `
  --fire-weight 6 `
  --output checkpoints/hunter_bc_pixel_v5
```

模型直接学习18类联合动作；日志仍把油门、转向、开火的边缘 loss/准确率和开火 precision/recall 拆开显示。`supervised.train` 的边采边训方式保留用于旧实验，但不再作为推荐入口。

先看人机本身：

```powershell
python -m supervised.watch --rows 6 --cols 6
```

看克隆模型打人机：

```powershell
python -m supervised.evaluate --checkpoint checkpoints/hunter_bc_pixel_v5/latest.pt --games 200 --no-render
```

### 2. 强化学习

监督差不多之后，再继承权重做 PPO。奖励只有胜负、自杀和超时。

```powershell
python -m rl.train `
  --opponent hunter `
  --initialize-from checkpoints/hunter_bc_pixel_v5/latest.pt `
  --teacher-coef 0 `
  --potential-scale 0.2 `
  --opponent-self-kill-reward 0 `
  --output checkpoints/RL_pixel_v5_hunter `
  --total-steps 300000
```

默认 PPO 使用 16 个环境、128 步 rollout、512 小批量、`1e-4` 退火学习率和 `0.02` target KL。训练日志会记录每种对手最近 100 局的独立战绩、近似 KL、clip fraction、explained variance 和动作比例。冻结模型对手在训练时按策略概率采样动作，不固定使用 argmax。

第一阶段固定打 hunter。主动击杀稳定提升后，才把该阶段冻结权重加入下一阶段的对手池，例如：

```powershell
python -m rl.train `
  --initialize-from checkpoints/RL_pixel_v5_hunter/latest.pt `
  --opponent hunter checkpoints/RL_pixel_v5_hunter/latest.pt `
  --opponent-weights 0.5 0.5 `
  --teacher-coef 0 `
  --potential-scale 0.2 `
  --opponent-self-kill-reward 0 `
  --output checkpoints/RL_pixel_v5_pool1 `
  --total-steps 300000
```

监督 checkpoint 只负责初始化；推荐 RL 使用 `--teacher-coef 0`，让后续策略完全由胜负目标改进。

观战：

```powershell
python -m rl.evaluate --checkpoint checkpoints/RL_pixel_v5_hunter/latest.pt --opponent hunter --games 200 --no-render
```

正式比较建议至少打 200 局。默认相邻两局复用同一地图并交换双方位置，结果会给出 95% 置信区间：

```powershell
python -m rl.evaluate --checkpoint checkpoints/RL_pixel_v5_hunter/latest.pt --opponent hunter --games 200 --seed 8000 --no-render
```

`--resume` 继续同一次训练；`--initialize-from` 只拷权重、步数从 0 开始。

每辆坦克输入完整的 `2×96×96` 像素图（墙体和有效区域），支持每局随机的 `6..12` 行列，每格对应8个像素。动态实体不栅格化：最多两个其他坦克和十五颗子弹都保留连续坐标，并按精确位置从墙体 CNN 特征图双线性取样。实体集合用共享 MLP 和 `sum+max` 汇总；策略使用单个18类联合动作头。模型无 A* 路点、墙距射线、注意力和 LSTM，共约14万参数。

观察和动作头均已更换，因此所有旧 checkpoint 和旧离线数据都不能与新版混用；监督学习必须重新采集并从随机初始化训练。

PPO 的任务奖励只有一次终局结果：胜利 `+1.0`，失败、自杀和超时均为 `-1.0`，不再按子弹飞行时间跨 rollout 回写多份击杀奖励。训练默认另加 `0.2 × (γΦ(s')-Φ(s))` 的势函数塑形；它由 A* 路径距离和有视线时的瞄准状态构成，折扣累计后只差初始状态常数，不能通过来回靠近刷分。具体参数集中在 `rl/environment.py` 的 `RewardConfig` 中。`checkpoints/` 已被 Git 忽略，不会误提交较大的模型文件。

## 视频速度测量工具

独立工具位于 `tools/video_speed_analyzer.py`，不依赖也不修改游戏核心。运行：

```powershell
python tools\video_speed_analyzer.py
```

使用顺序：打开视频；用“1 标定单位”框选边长为一个单位的正方形；用“2 测线速度”在两个不同帧点击物体位置；用“3 测角速度”在两个不同帧拖出物体朝向；或用“4 测距离”从物体一端拖到另一端，测量坦克长宽、子弹直径等静态尺寸。时间轴会显示速度测量的两个样本标记。
