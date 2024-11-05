from gymnasium.envs.registration import register, registry

def register_envs(): 
  if "Env_CircuitSys_v01" in registry: return
  register(id="Env_CircuitSys_v01", entry_point="env_gym.env:CircuitSys")


