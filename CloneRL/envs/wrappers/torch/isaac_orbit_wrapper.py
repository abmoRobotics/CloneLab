from skrl.envs.torch.wrappers import wrap_env


def wrap_env(env):
    """ Wraps the environment with the required wrappers.
    Right now it is based on wrap_env from skrl.envs.torch.wrappers, but may be changed in the future."""
    env = wrap_env(env)
    return env
