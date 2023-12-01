

## Step 1. train policy

def train():
    import gymnasium as gym
    from stable_baselines3 import PPO

    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    model.save("ppo_cartpole")

## Step 2. collect data

def collect_data():
    import gymnasium as gym
    from stable_baselines3 import PPO

    from CloneRL.collectors.torch import SequentialCollectorGym
    from CloneRL.collectors.torch.data_recorder import HDF5DataRecorder
    from CloneRL.utils import sb3_get_actions
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, verbose=1)
    model.load("ppo_cartpole.zip")

    recorder = HDF5DataRecorder(
        base_filename="ppo_cartpole",
        num_envs=1,
        env=env,
        max_rows=100000)

    collector = SequentialCollectorGym(
        env=env,
        model=model,
        recorder=recorder,
        predict_fn=sb3_get_actions,
        num_episodes=60000)

    collector.collect()

## Step 3. train imitation learning model

def train_imitation_learning():
    import torch
    import torch.nn as nn
    class MlpPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.device = "cuda:0"
            self.fc1 = nn.Linear(4, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x: torch.Tensor):
            x = x["observations"].cuda()
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.sigmoid(self.fc3(x))
            return x

    from CloneRL.algorithms.torch.imitation_learning.bc import \
        BehaviourCloning as BC
    from CloneRL.dataloader.hdf import HDF_DEFAULT_IL_MAPPER, HDF5Dataset
    from CloneRL.trainers.torch.sequential import SequentialTrainer as Trainer

    # Define path to HDF5 file and mapper for training and validation
    dataset = HDF5Dataset("ppo_cartpole_0.h5", HDF_DEFAULT_IL_MAPPER, min_idx=0, max_idx=5_000)
    dataset_val = HDF5Dataset("ppo_cartpole_0.h5", HDF_DEFAULT_IL_MAPPER, min_idx=5_000, max_idx=6_000)

    # Define model
    policy = MlpPolicy().to("cuda:0")

    # Choose
    agent = BC(policy = policy, cfg={})

    # Define the trainer
    trainer = Trainer(cfg={"batch_size": 200}, policy=agent, dataset=dataset, val_dataset=dataset_val)

    # Start training
    trainer.train(epoch=10)

    return policy

## Step 4. evaluate the model

def evaluate(policy):
    import gymnasium as gym
    import torch

    env = gym.make("CartPole-v1", render_mode="human")
    with torch.no_grad():
        obs, info = env.reset()
        for _ in range(10000):
            #print(obs)
            action = policy({"observations": torch.Tensor(obs).unsqueeze(0)})
            action = action.cpu().numpy()[0][0].astype(int)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            if done:
                obs, info = env.reset()



if __name__ == "__main__":
    #train()
    #collect_data()
    policy = train_imitation_learning()
    evaluate(policy)
