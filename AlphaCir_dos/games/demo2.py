from gymnasium.envs.registration import register, registry
from env_gym.__init__ import register_envs

if __name__ == "__main__":
    register_envs()
    import gymnasium as gym

    env = gym.make("Env_CircuitSys_v01")
    env.test()