# RLRoverLab Training

These scripts train CloneLab policies on datasets recorded by RLRoverLab. They
do not import or launch Isaac Lab directly.

Both legacy Isaac Lab HDF5 recordings and RLRoverLab compressed RGB-D v2 files
are supported. The default `--dataset_format auto` keeps legacy files on the
old loader and switches compressed files to the zero-duplication RGB-D loader.

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
  --dataset /workspace/rlroverlab/datasets/rover_rgbd_wvga_compressed_1000.hdf5

python Examples/rlroverlab/train_iql.py \
  --dataset /workspace/rlroverlab/datasets/rover_rgbd_wvga_compressed_1000.hdf5

python Examples/rlroverlab/train_bc_recurrent.py \
  --dataset /workspace/rlroverlab/datasets/rover_rgbd_wvga_compressed_1000.hdf5

python Examples/rlroverlab/train_iql_recurrent.py \
  --dataset /workspace/rlroverlab/datasets/rover_rgbd_wvga_compressed_1000.hdf5
```

The compressed loader decodes RGB to normalized `[0, 1]` tensors and depth to
meters using the file metadata (`recommended_rgb_scale` and
`recommended_depth_scale_m`). It reconstructs `next_obs` through
`index/next_obs_index`; no physical `next_obs` group is required in the file.

By default compressed visual decode runs on CUDA with `--compressed_decode_backend cuda`:
RGB JPEG uses torchvision's CUDA JPEG path backed by nvJPEG, and depth JP2 uses
NVIDIA nvImageCodec/nvJPEG2000. The CPU `pillow` backend is only for debugging
or machines without the NVIDIA decode stack.

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

## State-To-Visual DAgger

The DAgger workflow is implemented as a separate orchestration script, so the
existing BC and IQL entry points above stay unchanged. It trains an initial
DINO/DA recurrent BC student on the teacher dataset when no initial checkpoint
is supplied, then repeats collection, aggregation, retraining, and evaluation.

```bash
python scripts/run_dagger_dino_da_iterations.py \
  --run-context host \
  --rounds 3 \
  --base-dataset datasets/rover_dino_da3_512x288_400k.hdf5 \
  --output-root runs/dagger_dino_da_h256
```

To start from an already-trained BC student instead of training round zero:

```bash
python scripts/run_dagger_dino_da_iterations.py \
  --run-context host \
  --rounds 3 \
  --initial-checkpoint runs/dino-da-bc-rnn-h256/<run>/checkpoints/actor/best_model_epoch_4.pt
```

Inside the RLRoverLab container, run from the CloneLab checkout and switch the
context to direct container execution:

```bash
cd /workspace/clonelab
python scripts/run_dagger_dino_da_iterations.py \
  --run-context container \
  --rounds 3 \
  --initial-checkpoint runs/dino-da-bc-rnn-h256/<run>/checkpoints/actor/best_model_epoch_4.pt
```

Each DAgger round writes a new HDF5 shard under `--output-root/round_N`, a
dataset spec containing the base dataset plus all collected shards, and a
summary JSON/text file with collection and evaluation metrics.

The runner shows a top-level `tqdm` stage bar and streams collection/training
progress live while still writing per-stage logs. Use `--no-live-output` to keep
subprocess output in log files only, or `--no-progress` to hide the top-level
stage bar.
