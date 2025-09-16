# CloneLab Results Scoreboard

This scoreboard tracks the performance of student models trained using the CloneLab framework.

## Leaderboard

| Model Name       | Model Type | Algorithm | Environment         | Metric  | Score | SR    | Notes                         |
|------------------|------------|-----------|---------------------|---------|-------|-------|-------------------------------|
| IQL-Student-1    | Student    | IQL       | RoverNav-v0        | Reward  | 470   | 0.82  | Student submission            |
| BC-CartPole-1    | Student    | BC        | CartPole-v1        | Reward  | 450   | 0.90  | Gymnasium baseline           |
| IQL-Image-1      | Student    | IQL       | RoverNav-v0        | Reward  | 445   | 0.80  | With RGB+Depth input        |
| BC-Student-1     | Student    | BC        | RoverNav-v1        | Reward  | 440   | 0.78  | Custom dataset               |
| CQL-Student-1    | Student    | CQL       | RoverSim-v2        | Reward  | 395   | 0.79  | Conservative approach        |

## Top Performers by Algorithm

### IQL (Implicit Q-Learning)
1. **IQL-Student-1** - 470 reward (RoverNav-v0)
2. **IQL-Image-1** - 445 reward (RoverNav-v0)

### BC (Behavior Cloning)  
1. **BC-CartPole-1** - 450 reward (CartPole-v1)
2. **BC-Student-1** - 440 reward (RoverNav-v1)

### CQL (Conservative Q-Learning)
1. **CQL-Student-1** - 395 reward (RoverSim-v2)

## Environment Descriptions

- **RoverNav-v0**: Basic rover navigation task
- **RoverNav-v1**: Enhanced rover navigation with obstacles  
- **RoverSim-v2**: Full rover simulation environment
- **CartPole-v1**: Classic control task from Gymnasium

## Metrics

- **Reward**: Average episodic reward over 100 evaluation episodes
- **SR**: Success Rate - percentage of episodes that achieve the task objective
- **Score**: Primary performance metric (typically reward)

## Training Information

All models are trained using the CloneLab framework with the following general setup:
- Dataset collection via expert policies or human demonstrations
- Training using CloneLab algorithms (BC, IQL, CQL)
- Evaluation in the same environment as training
- Standard hyperparameters unless noted otherwise

## Submission Guidelines

To submit your results:

1. Train your model using CloneLab
2. Evaluate on the target environment (100+ episodes)
3. Add your result using the scoreboard manager:
   ```bash
   python scoreboards/scoreboard_manager.py add-result \
     --name "Your-Model-Name" \
     --algorithm "IQL" \
     --environment "RoverNav-v0" \
     --reward 485 \
     --success_rate 0.83 \
     --notes "Brief description of your approach"
   ```
4. Or manually add a row to the leaderboard table

## Competition and Recognition

- 🏆 **Top Score**: IQL-Student-1 (470 reward)
- 🎯 **Best Success Rate**: BC-CartPole-1 (0.90 SR)
- 📈 **Most Improved**: Track your progress over time

Good luck and happy training!