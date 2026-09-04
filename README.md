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
├─ rl_stage1/        第一关：空场近距离对位
├─ rl_stage3/        迷宫关
├─ supervised/       监督学习模块
│  ├─ train.py       规则导师纯行为克隆训练
│  ├─ evaluate.py    监督模型与导师评估
│  └─ teachers/      可扩展的导师策略
│     ├─ astar.py    稳定驾驶的 A* 导师
│     └─ weak_combat.py 会寻路、瞄准和开火的规则型弱人机
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

完整流程是：先模仿 A* 导师学会寻路，再模仿规则型弱人机学会瞄准和开炮，最后让强化学习脚本继承监督模型，继续优化实战策略。

开始前进入项目和 Python 环境：

```powershell
conda activate teacher
cd D:\bysq\tank_rl_sim
```

### 1. 使用 A* 导师训练基础寻路

A* 导师会为当前状态生成油门和转向标签。模型先学习稳定地沿迷宫路径靠近敌人，这一阶段不训练开火。

```powershell
python -m supervised.train --teacher astar --total-steps 200000 --rows 6 --cols 6 --output checkpoints/approach
```

训练结果保存在 `checkpoints/approach/latest.pt`，训练指标保存在 `checkpoints/approach/teacher_log.csv`。

### 2. 继承寻路模型并模仿弱人机

弱人机会沿 A* 路径接近敌人；出现无遮挡直射角度时会停车瞄准；还会主动扫描车头能够到达的一圈离散角度，预演直射和一次反弹弹道。当前地点没有安全射击角度时，它会沿 A* 路径寻找最近的直射格子，先移动到射击位置。预测弹道会先击中自己时不会开炮。它会同时提供油门、转向和开火三个监督标签。

监督训练始终由弱人机控制两辆坦克，模型只读取状态并学习弱人机给出的动作，不会在采集过程中接管坦克。建议将这一版纯行为克隆模型保存在新的 `checkpoints/weak_combat_bc/latest.pt`：

```powershell
python -m supervised.train --teacher weak-combat --initialize-from checkpoints/approach/latest.pt --total-steps 300000 --rows 6 --cols 6 --output checkpoints/weak_combat_bc
```

日志除了油门和转向准确率，还会记录：

- `fire_acc`：开火和不开火的总体分类准确率。
- `fire_precision`：模型决定开火时，有多少次符合导师的开火判断。
- `fire_recall`：导师要求开火时，模型有多少次真的选择开火。
- `fire_rate`：当前批次中导师给出开火标签的比例。
- `stop_rate`：当前批次中导师要求停车的比例。
- `forward_rate`：当前批次中导师要求前进的比例。

开火正样本默认拥有 4 倍分类权重，避免模型因为“不开炮”样本较多而退化成永远不开炮。

如果只想确认程序能否正常启动，可以把训练量改得很小：

```powershell
python -m supervised.train --teacher weak-combat --total-steps 32 --num-envs 2 --rollout-steps 8 --epochs 1 --minibatch-size 16 --output checkpoints/smoke
```

### 3. 查看弱人机和监督模型的效果

先看规则型弱人机本身（双方都由导师控制，不是 checkpoint）。有墙应只靠近，无墙应停车对准后直射。`R` 换图，`Esc` 退出：

```powershell
python -m supervised.watch --teacher weak-combat --rows 6 --cols 6
```

终端会大约每 0.5 秒打印油门/转向/开火；出现 `fire=1` 或子弹时说明导师在开炮。

也可以用评估入口连续打若干局并统计开火比例：

```powershell
python -m supervised.evaluate --teacher weak-combat --task combat --games 20
```

再加载监督模型，显示 20 局自博弈：

```powershell
python -m supervised.evaluate --checkpoint checkpoints/weak_combat_bc/latest.pt --task combat --games 20
```

只统计结果而不显示窗口：

```powershell
python -m supervised.evaluate --checkpoint checkpoints/weak_combat_bc/latest.pt --task combat --games 100 --no-render
```

评估结果会额外输出 `fire_command_rate`，可以直接检查模型是否选择过开炮。

原来的纯靠近模型仍可这样查看：

```powershell
python -m supervised.evaluate --checkpoint checkpoints/approach/latest.pt --task approach --games 20
```

### 4. 使用分关强化学习（推荐，不继承监督模型）

现在是两个独立包：

1. `rl_stage1`：空场近距离对位。
2. `rl_stage3`：随机迷宫。

第三关与第一关共用同一套局部墙图和奖励。迷宫关可以继承第一关权重：

```powershell
python -m rl_stage1.train --opponent idle --output checkpoints/combat_stage1_vs_idle
python -m rl_stage1.train --opponent move --initialize-from checkpoints/combat_stage1_vs_idle/latest.pt --output checkpoints/combat_stage1_vs_move
python -m rl_stage3.train --opponent move --initialize-from checkpoints/combat_stage1_full/latest.pt --output checkpoints/combat_stage3_maze
```

默认 `--opponent self` 是两边共用正在更新的策略。也可以让对手用规则脚本或一份冻结模型（只更新自己这边的轨迹）：

```powershell
python -m rl_stage1.train --opponent aim
python -m rl_stage1.train --opponent mix
python -m rl_stage1.train --opponent model --opponent-model checkpoints/combat_stage1_open_close/latest.pt
```

脚本有 `idle`（站桩）、`move`（会动但不开火）、`random`、`aim`（转向对准后开火）、`chase`（靠近并开火）、`dodge`（侧移躲弹）、`mix`（每局随机抽一种脚本）。

迷宫从零训练直接跑第三关，不要 `--initialize-from`：

```powershell
python -m rl_stage3.train
```

训练输出中的 `fire_rate` 是模型选择开火的决策比例。

### 5. 查看强化学习模型的效果

评估会默认使用 checkpoint 里保存的布局和出生方式。看第一关：

```powershell
python -m rl_stage1.evaluate --games 10
python -m rl_stage1.evaluate --games 10 --opponent aim
python -m rl_stage1.evaluate --games 10 --opponent model --opponent-model checkpoints/combat_stage1_open_close/step_10240.pt
```

看迷宫关：

```powershell
python -m rl_stage3.evaluate --games 10
```

只统计结果而不显示窗口：

```powershell
python -m rl_stage3.evaluate --games 100 --no-render
```

### 6. 从断点继续同一种训练

如果训练被提前停止，应使用 `--resume`，而不是 `--initialize-from`。`--resume` 会同时恢复模型参数、Adam 优化器状态和累计步数。

继续弱人机监督学习，例如从当前进度训练到累计 500000 步：

```powershell
python -m supervised.train --resume checkpoints/weak_combat_bc/latest.pt --total-steps 500000
```

继续某一关 PPO，对该关目录用 `--resume`：

```powershell
python -m rl_stage1.train --resume checkpoints/combat_stage1_open_close/latest.pt --total-steps 150000
```

继续迷宫 PPO：

```powershell
python -m rl_stage3.train --resume checkpoints/combat_stage3_maze/latest.pt --total-steps 400000
```

`--total-steps` 表示最终希望达到的累计步数，并不是本次额外增加的步数。不指定 `--output` 时，续训结果会继续保存在 checkpoint 所在目录。

简而言之：

- `--initialize-from`：继承某个模型的能力，开始一个新的训练阶段。
- `--resume`：恢复上次中断的位置，继续同一个训练阶段。

每辆坦克的输入包括 `1×48×48` 局部墙图、自身 12 维状态（无绝对坐标），以及其他坦克和子弹两个相对自身的可变长集合。CNN 处理墙图；坦克和子弹各自经过共享 MLP 再掩码平均。旧的 `5×128×128` 全图模型与当前结构不兼容，需要重新训练。

PPO 奖励目前只有胜负、超时、自杀和真正生成子弹时的小额 `fire_bonus`。被敌人击毁为 `-1.0`，被自己的子弹击毁为 `-0.1`。具体参数集中在各关 `environment.py` 的 `RewardConfig` 中。`checkpoints/` 已被 Git 忽略，不会误提交较大的模型文件。

## 视频速度测量工具

独立工具位于 `tools/video_speed_analyzer.py`，不依赖也不修改游戏核心。运行：

```powershell
python tools\video_speed_analyzer.py
```

使用顺序：打开视频；用“1 标定单位”框选边长为一个单位的正方形；用“2 测线速度”在两个不同帧点击物体位置；用“3 测角速度”在两个不同帧拖出物体朝向；或用“4 测距离”从物体一端拖到另一端，测量坦克长宽、子弹直径等静态尺寸。时间轴会显示速度测量的两个样本标记。
