# Student Result Template

Use this template when adding a new student result trained using CloneLab.

## Required Information

- **Model Name**: Unique identifier for your student model
- **Algorithm**: CloneLab algorithm used (IQL, BC, CQL)
- **Environment**: Target environment (e.g., RoverNav-v0, CartPole-v1)
- **Reward**: Average episodic reward over evaluation runs
- **Success Rate**: Percentage of successful episodes (0.0 to 1.0)
- **Notes**: Brief description of approach, special features, dataset info

## Example Entry

```
Model Name: IQL-Vision-v1
Algorithm: IQL
Environment: RoverNav-v0
Reward: 478.0
Success Rate: 0.84
Notes: Vision-based model with RGB+Depth input, custom reward shaping
```

## Using the CLI

```bash
python scoreboards/scoreboard_manager.py add-result \
  --name "IQL-Vision-v1" \
  --algorithm "IQL" \
  --environment "RoverNav-v0" \
  --reward 478 \
  --success-rate 0.84 \
  --notes "Vision-based model with RGB+Depth input, custom reward shaping"
```

## Training Tips

- Use the CloneLab framework for training
- Evaluate on at least 100 episodes for reliable statistics
- Document any special preprocessing or network architectures
- Include information about training time and computational resources