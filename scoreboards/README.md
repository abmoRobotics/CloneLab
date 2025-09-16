# CloneLab Scoreboards

This directory contains the standardized scoreboard system to compare the performance of baseline models (from RLRoverLab) and student models (trained using CloneLab algorithms like IQL, BC, CQL).

## Structure

- **`rlroverlab_baselines.md`** - Official baseline results from RLRoverLab
- **`clonelab_results.md`** - Student model results trained using CloneLab
- **`scoreboard_manager.py`** - Automated script for managing scoreboard entries
- **`templates/`** - Templates for adding new entries

## Quick Start

### Viewing Results
- See [RLRoverLab Baselines](rlroverlab_baselines.md) for official baseline scores
- See [CloneLab Results](clonelab_results.md) for student model scores

### Adding New Results
Use the scoreboard manager script:
```bash
python scoreboards/scoreboard_manager.py add-result --help
```

Or manually edit the respective markdown files following the existing format.

## Metrics

- **Reward**: Average episodic reward
- **SR**: Success Rate (percentage of successful episodes)  
- **Custom metrics**: Algorithm-specific or environment-specific metrics

## Environments

Common evaluation environments:
- **Rover environments**: Various rover navigation tasks
- **Gymnasium**: CartPole, MountainCar, etc.
- **Isaac Sim (Orbit)**: Robotics simulation environments
- **IsaacLab**: Advanced robotics environments

## Algorithms Supported

**Imitation Learning:**
- BC (Behavior Cloning)

**Offline RL:**
- IQL (Implicit Q-Learning)
- CQL (Conservative Q-Learning)