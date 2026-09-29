# rl_loco_lab → rl_mjlab_env 迁移说明

## 边界

本项目只迁移 `rl_loco_lab` 的 Unitree Go2 VAE+AMP 训练路径：

- 保留 Go2 rough-terrain VAE+AMP 任务；
- 不迁移 B2；
- 删除当前项目原有的纯 AMP 任务、算法入口和 train/play；
- 不迁移旧项目的通用 train/play、键盘控制和数据回放脚本；
- 使用 MjLab 官方 train/play 配置和 manager-based 环境接口。

## 包结构

| 职责 | 旧项目 | 当前项目 |
| --- | --- | --- |
| 自定义训练后端 | 仓库根目录 `rsl_rl_local/` | 独立 distribution `rsl_rl_local/` |
| 环境与任务 | `source/rl_loco_lab/.../vae_amp_loco` | `src/rl_mjlab_env/tasks/ampvae` |
| Go2 资产 | Isaac Lab USD 配置 | MjLab MJCF + OBJ package data |
| 启动配置 | Isaac Lab `configclass` | Python dataclass + MjLab registry |
| train/play | 旧项目自定义通用脚本 | 继承 MjLab 官方 CLI 配置的 AMPVAE 专用薄入口 |

`rsl_rl_local` 的算法、runner、storage、网络、AMP loader 和日志工具均从旧项目
迁入。为删除 `rsl-rl-lib==5.2.0`，原本依赖上游 `rsl_rl` 的 spatial-softmax
辅助网络改为包内自包含实现；其余核心 VAE+AMP 算法结构保持不变。

Go2 配置沿用 MjLab 官方任务结构：`config/go2/env_cfgs.py` 只包含环境 factory，
`config/go2/rl_cfg.py` 只包含训练配置，`config/go2/__init__.py` 负责注册任务。

## 训练语义对齐

| 项目 | 迁移后的值 |
| --- | --- |
| 控制周期 | physics dt `0.005`，decimation `4`，policy dt `0.02` |
| actor observation | 45 维 |
| estimator history | `5 × 45 = 225` 维 |
| critic observation | 48 维 |
| privileged observation | 43 维 |
| terrain height map | 187 维 |
| AMP observation | 39 维 |
| VAE ground truth | velocity 3 + COM 3 + mass 1 = 7 维 |
| actor input | actor 45 + VAE output 23 = 68 维 |
| critic input | critic 48 + privileged latent 16 + height latent 32 = 96 维 |
| observation lag | angular/gravity/velocity `0..2`，joint position `0..1` |
| actuator lag | `0..3` |
| PD gains | stiffness `25.0`，damping `0.5` |
| action scale | `0.25` |
| bad-orientation threshold | `1.0` rad |
| positive reward clamp | enabled |
| terrain curriculum | MjLab difficulty-row curriculum，初始 level `0..1` |
| undesired-contact history | 一个 policy step 内的 4 个 physics substep |
| joint-acceleration weight | `-1.0e-7`（MuJoCo `qacc` 后端校准值） |
| AMP preload | 2,000,000 transitions |
| training iterations | 100,000 |

旧项目的 VAE head、PPO 超参数、AMP discriminator、domain randomization、奖励
结构均保留。rough terrain 使用 MjLab 原生 preset，并显式打开
`terrain_generator.curriculum`；level 的升降使用 MjLab 新版
`terrain_levels_vel`，初始 level `0..1` 与旧项目一致。MJLab 的物理字段表示不同，
因此力矩限制、碰撞和传感器通过 MJCF/MjLab 等价配置表达，而不是复制
Isaac/PhysX 字段名。

Go2 MJCF 把 foot geom 挂在 calf body 下，因此不能像旧 USD 资产一样按 calf body
统计 undesired contact，否则正常足端着地也会被误判。迁移层改为只匹配 thigh/calf
collision geom，排除 foot geom，再把每条 calf 的两个 collision geom 聚合为一个
逻辑 link；接触力在一个 policy step 的 4 个 physics substep 上取最大值。关节加速度
保留 MuJoCo 直接提供的 `qacc`，它比 Isaac Lab 的速度有限差分更接近瞬时物理量；
根据匹配 rollout 的平方和量级，将该奖励权重从 `-2.5e-7` 校准为 `-1.0e-7`。

观测 delay buffer 每个 physics substep 推进，并在 episode reset 后采样一次 lag；
estimator 输入直接保存同一份 actor observation 的 5 帧时间序列（oldest → newest）。
这两点保持了旧 `VaeAmpRLEnv` 的时序语义，而不是采用按 term 展开的独立 history。

## MjLab 原生复用

可以直接复用的部分使用 MjLab 原生实现：任务 registry、scene/entity、action、
observation history/delay、reward/event/termination/curriculum manager、官方 CLI
配置、GPU 选择、checkpoint 查找、video wrapper 和 viewer。

没有强行套用官方 `MjlabOnPolicyRunner`。旧 VAE+AMP runner 同时维护 PPO、VAE、
AMP replay buffer、motion loader、discriminator optimizer 和 terminal AMP state，
这些并不是官方 runner 的一个小扩展点。保留独立 runner 能最大限度保证算法一致；
边界通过 `VaeAmpVecEnvWrapper` 对接官方环境接口。

## 有意保留的后端差异

- Isaac Lab/PhysX 替换为 MjLab/MuJoCo Warp。
- USD 资产替换为 MJCF 资产。
- frame transformer、height scanner、contact sensor 使用 MjLab sensor API。
- 物理随机化使用 MjLab DR API；随机变量范围和观测输出维度保持一致。
- MjLab manager 原生负责 observation history 和随机 lag。
