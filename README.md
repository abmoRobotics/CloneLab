# CloneLab
The purpose of this repository is to make a framework that can be used to train BC and Offline RL and evaluate them in isaac sim while training. The repository can be used for the following three things.
1. Collecting data using a premade policy
2. Train a student policy using BC or Offline RL
3. Evaluate the policy in the simulation framework of choice

## Scoreboards 🏆

CloneLab includes a comprehensive scoreboard system to track and compare model performance:

- **[RLRoverLab Baselines](scoreboards/rlroverlab_baselines.md)** - Official baseline results from RLRoverLab
- **[CloneLab Results](scoreboards/clonelab_results.md)** - Student model results trained using CloneLab

### Quick Start with Scoreboards
```bash
# View current results
python scoreboards/scoreboard_manager.py list

# Add your result
python scoreboards/scoreboard_manager.py add-result \
  --name "My-IQL-Model" \
  --algorithm "IQL" \
  --environment "RoverNav-v0" \
  --reward 480 \
  --success-rate 0.85 \
  --notes "My approach description"

# Export to CSV
python scoreboards/scoreboard_manager.py export --output results.csv
```

See [scoreboards/README.md](scoreboards/README.md) for detailed documentation.

# Examples
The following illustrates code snippets for each step.

## 1. Collect Data
The following code shows how to train an agent and use the agent to collect data
```python
from CloneRL.collectors.torch.data_recorder import HDF5DataRecorder
from CloneRL.collectors.torch import SequentialCollectorGym
from CloneRL.utils import sb3_get_actions

import gymnasium as gym
from stable_baselines3 import PPO

def main():
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, verbose=1)
    model.load("ppo_cartpole.zip")

    recorder = HDF5DataRecorder(
        base_filename="ppo_cartpole",
        num_envs=1,
        env=env,
        max_rows=10000)

    collector = SequentialCollectorGym(
        env=env,
        model=model,
        recorder=recorder,
        predict_fn=sb3_get_actions,
        num_episodes=1000)

    collector.collect()

if __name__ == "__main__":
    main()

```

## 2. Train Policy
```python
from CloneRL.algorithms.torch.imitation_learning.bc import BehaviourCloning as BC
from CloneRL.trainers.torch.sequential import SequentialTrainer as Trainer
from CloneRL.models.torch import MlpPolicy
from CloneRL.dataloader.hdf import HDF5Dataset, HDF_DEFAULT_IL_MAPPER

# Define path to HDF5 file and mapper for training and validation
data="/home/dataset_path"
dataset = HDF5Dataset(data, IL_MAPPER, min_idx=0, max_idx=100_000)
dataset_val = HDF5Dataset(data, IL_MAPPER, min_idx=100_000, max_idx=120_000)

# Define model
policy = MlpPolicy()

# Define agent based on BC algorithm
agent = BC(policy = policy, cfg={})

# Define the trainer with a batch size of 1000
trainer = Trainer(cfg={"batch_size": 1000}, policy=agent, dataset=dataset, val_dataset=dataset_val)

# Start training
trainer.train(epoch=30)
```

## 3. Evaluate Policy
The following code shows how the framework can be used to evaluate a policy
```python
from CloneRL.algorithms.torch.imitation_learning.bc import BehaviourCloning as BC
from CloneRL.trainers.torch.sequential import SequentialTrainer as Trainer
from CloneRL.models.torch import MlpPolicy

# Define model
policy = MlpPolicy()
policy.load_model("/home/dataset_path")

# Define agent based on BC algorithm
agent = BC(policy = policy, cfg={})

# Define the trainer
trainer = Trainer(policy=agent, cfg={})

# Load Orbit Environment
env = load_isaac_orbit_env(task_name="Cartpole-v0")
env = wrap_env(env)

# Evaluate
trainer.evaluate(env, num_steps=10000)
```
