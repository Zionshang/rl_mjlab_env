# rl_mjlab_env

```bash
conda create -n rl_mjlab_env python=3.10
conda activate rl_mjlab_env
pip install -e .
```

## Dataset

AMP motion data is not tracked by git. Put the Go2 NPZ files here:

```bash
mkdir -p dataset/unitree_go2/trot
ln -s /path/to/unitree_go2/trot/npz dataset/unitree_go2/trot/npz
```

## Commands

| Algorithm | Train | Play |
| --- | --- | --- |
| AMP | `train_amp Mjlab-AMP-Flat-Unitree-Go2 --gpu-ids [0] --env.scene.num-envs 4096 --video True --agent.max-iterations 10000` | `play_amp Mjlab-AMP-Flat-Unitree-Go2 --num-envs 4 --viewer viser --checkpoint-file path/to/checkpoint.pt` |
| Locomotion | `train_locomotion Mjlab-Locomotion-Rough-Unitree-Go2 --gpu-ids [0] --env.scene.num-envs 4096 --video True --agent.max-iterations 10000` | `play_locomotion Mjlab-Locomotion-Rough-Unitree-Go2 --num-envs 4 --viewer viser --checkpoint-file path/to/checkpoint.pt` |
