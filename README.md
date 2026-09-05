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

### 1. 模仿寻路开火人机（监督预训练）

两边都由人机开车，模型只学它的油门、转向、开火。人机会 A* 绕墙，看见敌人就打，有弹就躲。

```powershell
python -m supervised.train --total-steps 200000 --output checkpoints/hunter_bc
```

先看人机本身：

```powershell
python -m supervised.watch --rows 6 --cols 6
```

看克隆模型打人机：

```powershell
python -m supervised.evaluate --checkpoint checkpoints/hunter_bc/latest.pt --games 10
```

### 2. 强化学习

监督差不多之后，再继承权重做 PPO。奖励只有胜负、自杀和超时。

```powershell
python -m rl.train --opponent hunter --initialize-from checkpoints/hunter_bc/latest.pt --output checkpoints/rl --total-steps 200000
```

对手池可为每个条目指定权重。下面的例子用当前候选权重初始化新阶段，50% 对最近冻结版本、25% 对 hunter、其余 25% 对更早版本：

```powershell
python -m rl.train `
  --initialize-from "checkpoints/RL_对手池2_加入最新/latest.pt" `
  --opponent hunter "checkpoints/RL_开火归因_对人机/latest.pt" "checkpoints/RL_对手池_人机加自己/latest.pt" "checkpoints/RL_对手池2_加入最新/latest.pt" `
  --opponent-weights 0.25 0.10 0.15 0.50 `
  --output "checkpoints/RL_稳定训练1" `
  --total-steps 500000 --no-plot
```

默认 PPO 使用 16 个环境、128 步 rollout、512 小批量、`1e-4` 退火学习率和 `0.02` target KL。训练日志会记录每种对手最近 100 局的独立战绩、近似 KL、clip fraction、explained variance 和动作比例。冻结模型对手在训练时按策略概率采样动作，不固定使用 argmax。

如果混合对手池长期只在 50% 附近切换策略，先做固定 hunter 专项训练，不要继续扩大池子：

```powershell
python -m rl.train `
  --initialize-from "checkpoints/RL_稳定训练1/latest.pt" `
  --opponent hunter `
  --teacher-coef 0.03 `
  --potential-scale 0.2 `
  --output "checkpoints/RL_hunter专项" `
  --total-steps 300000
```

`teacher-coef` 只在智能体实际访问到的状态上加入一个随学习率衰减的 hunter 辅助分类损失，用来保住寻路、躲弹和开火基本功；PPO 仍由胜负决定改进方向。坦克和子弹集合使用零初始化的可学习注意力，加载旧 checkpoint 时初始行为与原来的平均池化一致。

观战：

```powershell
python -m rl.evaluate --checkpoint checkpoints/rl/latest.pt --opponent hunter --games 10
```

正式比较建议至少打 200 局。默认相邻两局复用同一地图并交换双方位置，结果会给出 95% 置信区间：

```powershell
python -m rl.evaluate --checkpoint "checkpoints/RL_稳定训练1/latest.pt" --opponent hunter --games 200 --no-render
```

`--resume` 继续同一次训练；`--initialize-from` 只拷权重、步数从 0 开始。

每辆坦克的输入包括 `1×48×48` 局部墙图、自身 12 维状态（无绝对坐标），以及其他坦克和子弹两个相对自身的可变长集合。

PPO 的任务奖励只有一次终局结果：胜利 `+1.0`，失败、自杀和超时均为 `-1.0`，不再按子弹飞行时间跨 rollout 回写多份击杀奖励。训练默认另加 `0.2 × (γΦ(s')-Φ(s))` 的势函数塑形；它由 A* 路径距离和有视线时的瞄准状态构成，折扣累计后只差初始状态常数，不能通过来回靠近刷分。具体参数集中在 `rl/environment.py` 的 `RewardConfig` 中。`checkpoints/` 已被 Git 忽略，不会误提交较大的模型文件。

## 视频速度测量工具

独立工具位于 `tools/video_speed_analyzer.py`，不依赖也不修改游戏核心。运行：

```powershell
python tools\video_speed_analyzer.py
```

使用顺序：打开视频；用“1 标定单位”框选边长为一个单位的正方形；用“2 测线速度”在两个不同帧点击物体位置；用“3 测角速度”在两个不同帧拖出物体朝向；或用“4 测距离”从物体一端拖到另一端，测量坦克长宽、子弹直径等静态尺寸。时间轴会显示速度测量的两个样本标记。
