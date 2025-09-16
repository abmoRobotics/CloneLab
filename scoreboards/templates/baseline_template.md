# Baseline Result Template

Use this template when adding a new baseline result from RLRoverLab.

## Required Information

- **Model Name**: Unique identifier for your baseline model
- **Algorithm**: Algorithm used (e.g., SAC, BC, CQL, AWAC, PPO)
- **Environment**: Target environment (e.g., RoverNav-v0, CartPole-v1)
- **Reward**: Average episodic reward over evaluation runs
- **Success Rate**: Percentage of successful episodes (0.0 to 1.0)
- **Notes**: Brief description, dataset used, special configurations

## Example Entry

```
Model Name: SAC-Baseline-v2
Algorithm: SAC
Environment: RoverNav-v1
Reward: 485.0
Success Rate: 0.88
Notes: Trained with improved hyperparameters, 1M steps
```

## Using the CLI

```bash
python scoreboards/scoreboard_manager.py add-baseline \
  --name "SAC-Baseline-v2" \
  --algorithm "SAC" \
  --environment "RoverNav-v1" \
  --reward 485 \
  --success-rate 0.88 \
  --notes "Trained with improved hyperparameters, 1M steps"
```