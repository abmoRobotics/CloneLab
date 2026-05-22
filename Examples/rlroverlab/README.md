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
