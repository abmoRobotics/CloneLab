from skrl.envs.torch.loaders import load_isaac_orbit_env


def load_isaac_orbit_env(task_name: str = ""):
    """ Load an environment from the Gym registry.
    Right now it is based on load_isaac_orbit_env from skrl.envs.torch.loaders, but may be changed in the future."""
    return load_isaac_orbit_env(task_name)
