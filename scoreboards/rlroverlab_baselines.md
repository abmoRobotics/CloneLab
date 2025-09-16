# RLRoverLab Baselines Scoreboard

This scoreboard tracks the performance of official baseline models from the RLRoverLab repository.

## Leaderboard

| Model Name    | Model Type | Algorithm | Environment         | Metric  | Score | SR    | Notes                    |
|---------------|------------|-----------|---------------------|---------|-------|-------|--------------------------|
| BC-Expert     | Baseline   | BC        | RoverNav-v0        | Reward  | 500   | 0.85  | Official baseline        |
| SAC-Baseline  | Baseline   | SAC       | RoverNav-v1        | Reward  | 480   | 0.87  | From RLRoverLab repo     |
| CQL-Baseline  | Baseline   | CQL       | RoverSim-v2        | Reward  | 420   | 0.81  | Default hyperparams      |
| AWAC-Baseline | Baseline   | AWAC      | RoverNav-v0        | Reward  | 465   | 0.84  | Trained on dataset X     |
| PPO-Expert    | Baseline   | PPO       | CartPole-v1        | Reward  | 475   | 0.95  | Gymnasium environment    |

## Environment Descriptions

- **RoverNav-v0**: Basic rover navigation task
- **RoverNav-v1**: Enhanced rover navigation with obstacles  
- **RoverSim-v2**: Full rover simulation environment
- **CartPole-v1**: Classic control task from Gymnasium

## Metrics

- **Reward**: Average episodic reward over 100 evaluation episodes
- **SR**: Success Rate - percentage of episodes that achieve the task objective
- **Score**: Primary performance metric (typically reward)

## Adding New Baselines

To add a new baseline result:

1. Run your baseline model evaluation
2. Use the scoreboard manager:
   ```bash
   python scoreboards/scoreboard_manager.py add-baseline \
     --name "Your-Model" \
     --algorithm "SAC" \
     --environment "RoverNav-v0" \
     --reward 450 \
     --success_rate 0.82 \
     --notes "Your description"
   ```
3. Or manually add a row to the table above following the same format

## Notes

- All baselines should be evaluated using the same evaluation protocol
- Results should be averaged over at least 100 episodes
- Include environment version and any special configuration in notes
- Contact RLRoverLab team for official baseline submissions