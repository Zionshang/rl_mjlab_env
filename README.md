# rl_loco_mj

locomotion 任务在 MjLab 上的版本

## 安装

```bash
conda create -n rl_loco_mj python=3.13
conda activate rl_loco_mj
pip install -e ./rsl_rl_local -e .
```

## 数据集

训练读取：

```text
dataset/unitree_go2/trot/npz/*.npz
```

数据集被 `.gitignore` 排除, 在新的 clone 中需要自行复制或链接：

```bash
mkdir -p dataset/unitree_go2/trot
cp -a /path/to/rl_loco_lab/source/rl_loco_lab/assets/datasets/go2/. \
  dataset/unitree_go2/trot/
```

## 训练与播放

train
```bash
train_ampvae Mjlab-AMPVAE-Rough-Unitree-Go2 \
  --gpu-ids '[0]' \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 100000 \
  --video True \
  --video-length 250 \
  --video-interval 2400
```

play
```
play_ampvae Mjlab-AMPVAE-Rough-Unitree-Go2 \
  --viewer viser \
  --checkpoint-file path/to/model.pt
```
