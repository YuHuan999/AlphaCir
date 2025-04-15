import copy

import ray
import torch

from models import AlphaCirNetwork, UniformNetwork
#################################################################
####################### Shared_Storage-Start- #######################
def make_uniform_network(num_actions: int) -> UniformNetwork:
    return UniformNetwork(num_actions)

class SharedStorage(object):
  """Controls which network is used at inference."""

  def __init__(self, num_actions: int):
    self._num_actions = num_actions
    self._networks = {}

  def latest_network(self) -> AlphaCirNetwork:
    if self._networks:
      return self._networks[max(self._networks.keys())]
    else:
      # policy -> uniform, value -> 0, reward -> 0
      return make_uniform_network(self._num_actions)

  def save_network(self, step: int, network: AlphaCirNetwork):
    self._networks[step] = network




#################################################################
####################### Shared_Storage- End - #######################

















@ray.remote
class SharedStorage:
    """
    Class which run in a dedicated thread to store the network weights and some information.
    """

    def __init__(self, checkpoint, config):
        self.config = config
        self.current_checkpoint = copy.deepcopy(checkpoint)

    def save_checkpoint(self, path=None):
        if not path:
            path = self.config.results_path / "model.checkpoint"

        torch.save(self.current_checkpoint, path)

    def get_checkpoint(self):
        return copy.deepcopy(self.current_checkpoint)

    def get_info(self, keys):
        if isinstance(keys, str):
            return self.current_checkpoint[keys]
        elif isinstance(keys, list):
            return {key: self.current_checkpoint[key] for key in keys}
        else:
            raise TypeError

    def set_info(self, keys, values=None):
        if isinstance(keys, str) and values is not None:
            self.current_checkpoint[keys] = values
        elif isinstance(keys, dict):
            self.current_checkpoint.update(keys)
        else:
            raise TypeError
