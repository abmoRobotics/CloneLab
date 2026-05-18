# RLRoverLab Training

These scripts train CloneLab policies on datasets recorded by RLRoverLab. They
do not import or launch Isaac Lab directly.

RLRoverLab remains responsible for:

- Isaac Sim / Isaac Lab runtime
- task registration
- dataset recording
- live policy evaluation

CloneLab remains responsible for:

- BC/IQL algorithms
- recurrent BC/IQL training
- model construction
- offline dataset loading
- checkpoint production

## Train

```bash
python Examples/rlroverlab/train_bc.py \
  --dataset /workspace/rlroverlab/datasets/rover_rgbd_expert.hdf5

python Examples/rlroverlab/train_iql.py \
  --dataset /workspace/rlroverlab/datasets/rover_rgbd_expert.hdf5

python Examples/rlroverlab/train_bc_recurrent.py \
  --dataset /workspace/rlroverlab/datasets/rover_rgbd_expert.hdf5

python Examples/rlroverlab/train_iql_recurrent.py \
  --dataset /workspace/rlroverlab/datasets/rover_rgbd_expert.hdf5
```

## Train Then Evaluate In RLRoverLab

```bash
python Examples/rlroverlab/train_iql.py \
  --dataset /workspace/rlroverlab/datasets/rover_rgbd_expert.hdf5 \
  --eval_after_train \
  --eval_num_envs 8 \
  --eval_steps 1000
```

If running inside the RLRoverLab container, the default eval command is:

```bash
/isaac-sim/python.sh /workspace/rlroverlab/examples/04_clonelab/eval_policy.py
```

Outside the container, the default eval command is:

```bash
docker exec rover-lab-base /isaac-sim/python.sh /workspace/rlroverlab/examples/04_clonelab/eval_policy.py
```

Override this with `--rlroverlab_eval_cmd` when needed.
