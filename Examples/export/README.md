# Export

Policy export now lives in `CloneRL.export`.

```bash
python -m CloneRL.export onnx \
  --checkpoint runs/<project>/<run>/checkpoints \
  --output exported/policy

python -m CloneRL.export tensorrt \
  --onnx exported/policy/policy.onnx \
  --output exported/policy/policy.engine
```
