# RLRoverLab Baselines Scoreboard

This scoreboard tracks the performance of official baseline models from the RLRoverLab repository.

## Leaderboard

| Model Name    | Model Type | Algorithm | Environment         | Metric  | Score | SR    | Notes                    |
|---------------|------------|-----------|---------------------|---------|-------|-------|--------------------------|
| PPO-Teacher   | Teacher    | PPO       | RoverNav-v0        | Reward  | 500   | 0.85  | Official baseline        |

## Environment Descriptions

- **RoverNav-v0**: Basic rover navigation task
- **RoverNav-v1**: Enhanced rover navigation with obstacles  
- **RoverSim-v2**: Full rover simulation environment

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
