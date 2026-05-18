# RLRoverLab Run Configs

Each training script loads its matching default YAML from this directory.
Passing `--config my_run.yaml` deep-merges your file over that default.
Explicit CLI flags still override both YAML files.

Example override:

```yaml
dataset:
  path: /data/rover_train.hdf5

algorithm:
  tau: 0.02
  expectile: 0.9
```

Run it with:

```bash
python Examples/rlroverlab/train_iql.py --config my_run.yaml
```

Model configs can be overridden inline:

```yaml
models:
  actor:
    config:
      mlp_features: [256, 128, 64]
```
