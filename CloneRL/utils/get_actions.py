from stable_baselines3.common.base_class import BaseAlgorithm

def sb3_get_actions(model: BaseAlgorithm, obs):
    return model.predict(obs)[0]