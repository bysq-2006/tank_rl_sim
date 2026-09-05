# 课程强化学习模块

本目录使用一套模型和 PPO，通过逐渐提高地图、出生点和对手难度从零训练，不需要监督预训练。

```text
rl/
├─ model.py          模型及旧导入路径
├─ observation.py    可观测输入编码
├─ curriculum/       阶段配置、抽样和晋级
├─ envs/             RL环境及地图/出生场景
├─ opponents/        脚本对手、历史模型池及冻结批量推理
├─ rewards/          击杀、死亡、终局和势函数奖励
├─ training/         rollout、PPO、检查点和训练入口
├─ evaluation/       固定种子确定性评估
└─ tests/            RL局部测试
```

## 开始训练

```powershell
conda activate teacher
cd D:\bysq\tank_rl_sim
python -m rl.training.train --output checkpoints\tank_rl_curriculum_v2 --total-steps 3000000
```

程序默认从阶段 0 开始。每次晋级会保留同一个模型的权重，并将课程状态与优化器一起保存到检查点。继续训练：

```powershell
python -m rl.training.train `
  --resume checkpoints\tank_rl_curriculum_v2\latest.pt `
  --output checkpoints\tank_rl_curriculum_v2 `
  --additional-steps 1000000
```

如果输出目录已经有 `latest.pt`，程序会拒绝无意覆盖；此时必须使用 `--resume`，或者换一个新的输出目录。

训练默认每 10 次 PPO 更新在 `previews` 子目录保存一份可直接观战的模型，并滚动保留最近 20 份。预览文件不含优化器，因此比完整续训检查点更小；`latest.pt` 仍然负责续训。可以调整保存频率和保留数量：

```powershell
python -m rl.training.train `
  --resume checkpoints\tank_rl_curriculum_v2\latest.pt `
  --output checkpoints\tank_rl_curriculum_v2 `
  --additional-steps 500000 `
  --preview-interval 5 `
  --max-preview-checkpoints 30
```

观看某个中间版本时，把预览文件直接传给观战脚本：

```powershell
python -m rl.evaluation.watch `
  --checkpoint checkpoints\tank_rl_curriculum_v2\previews\preview_update_000060_step_000000122880.pt `
  --opponent chaser `
  --games 0
```

### 实时训练图表

训练命令启动后会默认弹出独立的实时折线图窗口，展示训练奖励、训练与评估胜率、超时率、策略/价值损失、策略熵、KL、平均开炮数、危险开炮数、历史对手占比、当前关卡和学习率。关闭图表窗口不会停止训练。

每轮原始指标都会追加到输出目录的 `training_metrics.jsonl`；每次保存预览模型时还会更新 `training_dashboard.png`。续训会读取检查点之前的历史指标并接着绘制。最多在窗口显示最近 1000 个点，可以修改或关闭窗口：

```powershell
# 窗口显示最近 2000 个更新点
python -m rl.training.train --output checkpoints\new_run --dashboard-points 2000 --total-steps 3000000

# 无图形界面的机器上禁用弹窗，但仍记录JSONL指标
python -m rl.training.train --output checkpoints\new_run --no-dashboard --total-steps 3000000
```

手动从第 3 阶段开始，但仍允许之后自动晋级：

```powershell
python -m rl.training.train --stage 3 --output checkpoints\stage3_auto --total-steps 3000000
```

如果要从第三关开始、完全固定使用简单闪避敌人，先使用固定关卡训练：

```powershell
python -m rl.training.train `
  --stage 3 `
  --fixed-stage `
  --output checkpoints\stage3_dodger `
  --total-steps 1000000
```

从第三关开始自动晋级时，课程抽样会自动排除第 0～2 关的 `idle` 敌人；第 3 关及之后只会混合闪避、移动、射击和历史模型对手。

只训练第 3 阶段，不混入旧阶段、下一阶段，也不自动晋级：

```powershell
python -m rl.training.train --stage 3 --fixed-stage --output checkpoints\stage3_fixed --total-steps 3000000
```

续训时会默认恢复检查点中的阶段。也可以覆盖阶段，或者把锁定训练切回自动课程：

```powershell
python -m rl.training.train `
  --resume checkpoints\tank_rl_curriculum_v2\latest.pt `
  --stage 4 `
  --auto-curriculum `
  --additional-steps 1000000
```

## 独立评估

```powershell
python -m rl.evaluation.evaluate `
  --checkpoint checkpoints\tank_rl_curriculum_v2\latest.pt `
  --stage 0 `
  --games 200
```

## 使用渲染器观战

最新模型对追踪射击脚本：

```powershell
python -m rl.evaluation.watch `
  --checkpoint checkpoints\tank_rl_curriculum_v2\latest.pt `
  --opponent chaser `
  --stage 6
```

最新模型对另一个模型：

```powershell
python -m rl.evaluation.watch `
  --checkpoint checkpoints\tank_rl_curriculum_v2\latest.pt `
  --opponent checkpoints\old_run\latest.pt `
  --stage 6
```

脚本对手可选 `idle`、`random_mover`、`dodger`、`weak_shooter` 和 `chaser`。`dodger` 会追踪、瞄准并低频开火，同时在发现预计一秒内会命中的子弹时做简单侧向闪避；它只在无遮挡时开火，减少对墙自毁。省略 `--stage` 时读取主检查点保存的当前阶段；`--games 0` 持续播放，空格、回车或 `N` 跳过当前局，`Esc` 退出。

如果想让观战对手按照训练该关卡时的配置自动选择，使用 `--opponent training`。它会使用该关卡的脚本类型和开火概率，并按历史模型概率从同一训练目录的对手池中抽取旧模型（如果池中已有模型）：

```powershell
python -m rl.evaluation.watch `
  --checkpoint checkpoints\tank_rl_curriculum_v2\latest.pt `
  --stage 3 `
  --opponent training `
  --games 0
```

## 历史模型对手

训练会自动扫描输出目录中的 `stage_*.pt`，并把晋级时的冻结模型加入对手池。从第 1 关开始，每局在“该关脚本策略”和“历史模型”之间抽取。第 1 至第 6 关抽到历史模型的概率依次为 `15%`、`25%`、`35%`、`45%`、`55%`、`65%`，较新的已完成阶段会获得更高抽样权重。同一个快照在并行环境中的动作会合并成一次模型推理。

如果从第 1 关或更后面的检查点续训，但旧目录没有保留任何阶段模型，程序会先冻结一份“续训起点模型”作为首个历史对手，因此历史对战会立即生效。

每 200 个 PPO 更新还会把当前策略保存到 `opponent_pool`，防止后期只会打某一个固定旧版本。可调整或关闭：

```powershell
python -m rl.training.train `
  --resume checkpoints\tank_rl_curriculum_v2\latest.pt `
  --additional-steps 1000000 `
  --opponent-snapshot-interval 300

# 完全禁用历史模型，只使用脚本对手
python -m rl.training.train --no-historical-opponents --output checkpoints\scripts_only --total-steps 3000000
```

日志中的 `本轮历史模型对局数` 是本次采样内结束、且敌方由冻结旧模型控制的局数。第 0 关不使用历史模型，并已改为远距离、随机朝向的开放靶场，避免贴脸开火的固定捷径。

## 奖励原则

- 最终胜利 `+1`；失败、双方死亡和超时 `-1`。
- 前期主动击杀最多额外 `+0.30`，自身死亡额外 `-0.30`。
- 不再因为普通开炮本身给奖励；冷却或弹数已满时反复按键也不奖励。
- 发射前会按真实墙体反弹轨迹预演：预测会撞回自己的子弹，该发射扣 `0.03`。日志中的 `本轮危险开炮数` 记录这类开火。
- 如果敌人用自己的子弹把自己打死，该局标记为“敌方自毁”，玩家不会得到胜利奖励，也不会额外扣分；评估时不会计入胜率，避免转圈诱导敌人自毁形成高分捷径。
- 第 3～6 关的自身被击毁即时惩罚分别为 `-0.35`、`-0.30`、`-0.25`、`-0.20`，再叠加失败终局的 `-1.0`，让躲避子弹比无脑冒险更重要。
- 即时奖励和势函数奖励随阶段逐渐减小，最终完整脚本阶段分别降到 `0.05`。
- PPO 使用 `gamma=0.995`、`GAE lambda=0.99` 将延迟命中奖励向前传递；约 3 秒前的动作仍能收到约 58% 的信用，约 5 秒前仍有约 40%。
- 子弹发射者只用于训练奖励判断，不进入模型观察。
- 模型始终只接收未来能够从真实画面恢复的地图、坐标、朝向及子弹速度。

## 局部测试

```powershell
python -m pytest -q rl\tests\test_curriculum.py
```
