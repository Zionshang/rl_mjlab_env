# rl_mjlab_env

`rl_loco_lab` 的 Go2 VAE+AMP 任务在 MjLab 上的迁移版本。环境、观测、奖励、
随机化、网络和算法配置尽量保持原样，主要变化是仿真/任务后端从 Isaac Lab
替换为 MjLab。

## 安装

项目包含两个独立的 Python distribution：

- `rsl-rl-lib-local`：原 `rl_loco_lab/rsl_rl_local` 训练后端。
- `rl-mjlab-env`：MjLab 环境、Go2 资产、任务配置和 train/play 入口。

在空虚拟环境中应同时把两者交给 pip：

```bash
conda create -n rl_loco_mj python=3.13
conda activate rl_loco_mj
pip install -e ./rsl_rl_local -e .
```

不要只运行 `pip install -e .`：主包声明依赖独立 distribution
`rsl-rl-lib-local==3.0.1`，它不是 PyPI 包。原来的
`rsl-rl-lib==5.2.0` 依赖已完全移除。

## 数据集

训练读取：

```text
dataset/unitree_go2/trot/npz/*.npz
```

数据集被 `.gitignore` 排除。当前迁移工作区已从旧项目复制 343 个 NPZ 和 343 个
YAML；在新的 clone 中需要自行复制或链接：

```bash
mkdir -p dataset/unitree_go2/trot
cp -a /path/to/rl_loco_lab/source/rl_loco_lab/assets/datasets/go2/. \
  dataset/unitree_go2/trot/
```

## 训练与播放

只注册 Go2 VAE+AMP 任务；纯 AMP 任务和通用 train/play 没有迁移。

```bash
train_ampvae Mjlab-AMPVAE-Rough-Unitree-Go2 \
  --gpu-ids '[0]' \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 100000 \
  --video True \
  --video-length 250 \
  --video-interval 2400

play_ampvae Mjlab-AMPVAE-Rough-Unitree-Go2 \
  --viewer viser \
  --checkpoint-file path/to/model.pt
```

训练视频保存在
`logs/rsl_rl_local/vae_amp_go2/<run>/videos/train/`。`video-length` 和
`video-interval` 的单位都是 policy step；默认每次迭代采集 24 个 policy step，
所以 `--video-interval 2400` 相当于每 100 次训练迭代触发一次（并在第 0 步立即
触发第一次录制）。启动日志出现 `Recording training videos` 才表示视频参数已被
当前进程接收。

train/play 的 CLI 配置分别继承 MjLab 官方 `TrainConfig` 和 `PlayConfig`；环境
适配器继承官方 `ManagerBasedRlEnv`，向量环境适配器继承官方
`RslRlVecEnvWrapper`。自定义部分只负责旧 VAE+AMP runner 需要的 AMP 终止状态、
动作历史和 discriminator reward 注入。

迁移取舍和逐项对应关系见 [迁移说明](docs/migration.md)。
