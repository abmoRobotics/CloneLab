# CloneRL Export

Export is intentionally small:

```text
checkpoint + export_config.json -> ONNX -> optional TensorRT engine
```

Training scripts should write `export_config.json` into the checkpoint
directory. Then ONNX export only needs the checkpoint directory and an output
directory:

```bash
python -m CloneRL.export onnx \
  --checkpoint runs/<project>/<run>/checkpoints \
  --output exported/rover_policy
```

For recurrent policies, ONNX exposes hidden state explicitly:

```text
image/depth/proprioceptive + hidden_in -> action + hidden_out
```

Build TensorRT from the ONNX file in the target runtime:

```bash
python -m CloneRL.export tensorrt \
  --onnx exported/rover_policy/policy.onnx \
  --output exported/rover_policy/policy.engine \
  --fp16
```

TensorRT engines are specific to the CUDA/TensorRT/GPU runtime that builds
them. Keep ONNX as the portable artifact.
