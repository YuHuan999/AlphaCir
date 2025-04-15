import copy
import time

import numpy
import ray

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import AlphaCirNetwork, NetworkOutput

from games.CirsysGame import Game
from games.CirsysGame import AlphaCirConfig

from shared_storage import SharedStorage
from replay_buffer import ReplayBuffer

from typing import Any, Sequence
from collections import namedtuple
#################################################################
######################## Trainer -Start- #######################



def train_network(config, storage, replay_buffer):
    """Trains the network on data stored in the replay buffer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    network = AlphaCirNetwork(config.hparams_r, config.hparams_p, config.task_spec).to(device)
    target_network = AlphaCirNetwork(config.hparams_r, config.hparams_p, config.task_spec).to(device)
    target_network.load_state_dict(network.state_dict())  # 初始化目标网络

    optimizer = torch.optim.SGD(network.parameters(), lr=config.lr_init, momentum=config.momentum)

    for i in range(config.training_steps):
        if i % config.checkpoint_interval == 0:
            storage.save_network(i, network)

        if i % config.target_network_interval == 0:
            target_network.load_state_dict(network.state_dict())

        batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
        optimizer.zero_grad()
        loss = compute_loss(network, target_network, batch, device)
        loss.backward()
        optimizer.step()

    storage.save_network(config.training_steps, network)


def compute_loss(network, target_network, batch, device):
    total_loss = 0.0

    for observation, bootstrap_obs, target in batch:
        # Move tensors to device
        observation = move_to_device(observation, device)
        bootstrap_obs = move_to_device(bootstrap_obs, device)
        target_fidelity, target_length, target_policy, bootstrap_discount = target

        # Forward pass
        predictions = network(observation)
        bootstrap_predictions = target_network(bootstrap_obs)

        # Unpack predictions
        policy_logits = predictions["policy"]
        fidelity_logits = predictions["fidelity_value_logits"]
        length_logits = predictions["length_value_logits"]

        # Target for value（加上 TD bootstrapping）
        bootstrap_target = bootstrap_predictions["fidelity_value_logits"].detach()
        # target_fidelity = reward[1] + γ * reward[2] + γ² * reward[3] + γ³ * V(s_4)
        target_fidelity += bootstrap_discount * bootstrap_target
        
        policy_loss = F.cross_entropy(policy_logits, target_policy)
        fidelity_loss = scalar_loss(fidelity_logits, target_fidelity, network)
        length_loss = scalar_loss(length_logits, target_length, network)

        total_loss += policy_loss + fidelity_loss + length_loss

    return total_loss / len(batch)

def scalar_loss(pred_logits, scalar_target, network):
    """
    pred_logits: shape (B, num_bins)
    scalar_target: shape (B,)
    """
    two_hot_target = network.prediction.support.scalar_to_two_hot(scalar_target)
    log_probs = F.log_softmax(pred_logits, dim=-1)
    loss = -(two_hot_target * log_probs).sum(dim=-1).mean()
    return loss

def move_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [move_to_device(x, device) for x in data]
    else:
        return data













#################################################################
######################## Trainer - End - #######################



@ray.remote
class Trainer:
    """
    Class which run in a dedicated thread to train a neural network and save it
    in the shared storage.
    """

    def __init__(self, initial_checkpoint, config):
        self.config = config

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Initialize the network
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(copy.deepcopy(initial_checkpoint["weights"]))
        self.model.to(torch.device("cuda" if self.config.train_on_gpu else "cpu"))
        self.model.train()

        self.training_step = initial_checkpoint["training_step"]

        if "cuda" not in str(next(self.model.parameters()).device):
            print("You are not training on GPU.\n")

        # Initialize the optimizer
        if self.config.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.lr_init,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.lr_init,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise NotImplementedError(
                f"{self.config.optimizer} is not implemented. You can change the optimizer manually in trainer.py."
            )

        if initial_checkpoint["optimizer_state"] is not None:
            print("Loading optimizer...\n")
            self.optimizer.load_state_dict(
                copy.deepcopy(initial_checkpoint["optimizer_state"])
            )

    def continuous_update_weights(self, replay_buffer, shared_storage):
        # Wait for the replay buffer to be filled
        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:
            time.sleep(0.1)

        next_batch = replay_buffer.get_batch.remote()
        # Training loop

        while self.training_step < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            index_batch, batch = ray.get(next_batch)
            ## why so many information in batch?
            ## 这里 了解batch的构成
            next_batch = replay_buffer.get_batch.remote()
            ## updata network
            self.update_lr()
            (
                priorities,
                total_loss,
                value_loss,
                reward_loss,
                policy_loss,
            ) = self.update_weights(batch)

            if self.config.PER:
                # Save new priorities in the replay buffer (See https://arxiv.org/abs/1803.00933)
                replay_buffer.update_priorities.remote(priorities, index_batch)

            # Save to the shared storage
            if self.training_step % self.config.checkpoint_interval == 0:
                shared_storage.set_info.remote(
                    {
                        "weights": copy.deepcopy(self.model.get_weights()),
                        "optimizer_state": copy.deepcopy(
                            models.dict_to_cpu(self.optimizer.state_dict())
                        ),
                    }
                )
                if self.config.save_model:
                    shared_storage.save_checkpoint.remote()
            
            ## save target network to the shared storage
            if self.training_step % self.config.target_interval == 0:
                shared_storage.set_info.remote(
                    {
                        "_weights": copy.deepcopy(self.model.get_weights()),
                        "optimizer_state": copy.deepcopy(
                            models.dict_to_cpu(self.optimizer.state_dict())
                        ),
                    }
                )

            
            shared_storage.set_info.remote(
                {
                    "training_step": self.training_step,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "total_loss": total_loss,
                    "value_loss": value_loss,
                    "reward_loss": reward_loss,
                    "policy_loss": policy_loss,
                }
            )

            # Managing the self-play / training ratio
            if self.config.training_delay:
                time.sleep(self.config.training_delay)
            if self.config.ratio:
                while (
                    self.training_step
                    / max(
                        1, ray.get(shared_storage.get_info.remote("num_played_steps"))
                    )
                    > self.config.ratio
                    and self.training_step < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.5)

    def update_weights(self, batch):
        """
        Perform one training step.
        """

        (
                observation_batch,
                action_batch,
                target_correctness_values,
                target_length_values,
                target_policies,
                weight_batch,
                gradient_scale_batch,

        ) = batch

        ## ??
        # Keep values as scalars for calculating the priorities for the prioritized replay
        target_correctness_values_scalar = numpy.array(target_correctness_values, dtype="float32")
        priorities = numpy.zeros_like(target_correctness_values_scalar)

        device = next(self.model.parameters()).device
        if self.config.PER:
            weight_batch = torch.tensor(weight_batch.copy()).float().to(device)
        observation_batch = (torch.tensor(numpy.array(observation_batch)).float().to(device))
        action_batch = torch.tensor(action_batch).long().to(device).unsqueeze(-1)
        target_correctness_values = torch.tensor(target_correctness_values).float().to(device)
        target_length_values = torch.tensor(target_length_values).float().to(device)
        target_policies = torch.tensor(target_policies).float().to(device)
        gradient_scale_batch = torch.tensor(gradient_scale_batch).float().to(device)
        
        target_value_scalar = target_correctness_values + target_length_values ## for priorities
        # observation_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps+1, 1 (unsqueeze)
        # target_value: batch, num_unroll_steps+1
        # target_reward: batch, num_unroll_steps+1
        # target_policy: batch, num_unroll_steps+1, len(action_space)
        # gradient_scale_batch: batch, num_unroll_steps+1

        target_correctness_values = models.scalar_to_support_simply(target_correctness_values, self.config.support_size)
        target_length_values = models.scalar_to_support_simply(
            target_length_values, self.config.support_size
        )
        # target_value: batch, num_unroll_steps+1, 2*support_size+1
        # target_reward: batch, num_unroll_steps+1, 2*support_size+1

        ## Generate predictions
        value, correctness_value_logits, length_value_logits, policy_logits = self.model.initial_inference(
            observation_batch
        )
        predictions = [(correctness_value_logits, length_value_logits, policy_logits)]

        # for i in range(1, action_batch.shape[1]):
        #     value, reward, policy_logits, hidden_state = self.model.recurrent_inference(
        #         hidden_state, action_batch[:, i]
        #     )
        #     # Scale the gradient at the start of the dynamics function (See paper appendix Training)
        #     hidden_state.register_hook(lambda grad: grad * 0.5)
        #     predictions.append((value, reward, policy_logits))
        # predictions: num_unroll_steps+1, 3, batch, 2*support_size+1 | 2*support_size+1 | 9 (according to the 2nd dim)

        ## Compute losses
        correctness_loss, length_loss, policy_loss = (0, 0, 0)
        # correctness_predict_logits, length_predict_logits, policy_predict_logits = predictions[0]
        # # Ignore reward loss for the first batch step
        # current_value_loss, _, current_policy_loss = self.loss_function(
        #     value.squeeze(-1),
        #     reward.squeeze(-1),
        #     policy_logits,
        #     target_value[:, 0],
        #     target_reward[:, 0],
        #     target_policy[:, 0],
        # )
        # value_loss += current_value_loss
        # policy_loss += current_policy_loss
        # # Compute priorities for the prioritized replay (See paper appendix Training)
        # pred_value_scalar = (
        #     models.support_to_scalar(value, self.config.support_size)
        #     .detach()
        #     .cpu()
        #     .numpy()
        #     .squeeze()
        # )
        # priorities[:, 0] = (
        #     numpy.abs(pred_value_scalar - target_value_scalar[:, 0])
        #     ** self.config.PER_alpha
        # )

        for i in range(0, len(predictions)):
            correctness_predict_logits, length_predict_logits, policy_logits = predictions[i]
            (
                current_correctness_loss,
                current_length_loss,
                current_policy_loss,
            ) = self.loss_function(
                correctness_predict_logits(-1),
                length_predict_logits(-1),
                policy_logits,
                target_correctness_values[:, i],
                target_length_values[:, i],
                target_policies[:, i],
            )

            # Scale gradient by the number of unroll steps (See paper appendix Training)
            current_correctness_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i]
            )
            current_length_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i]
            )
            current_policy_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i]
            )

            correctness_loss += current_correctness_loss
            length_loss += current_length_loss
            policy_loss += current_policy_loss

            # Compute priorities for the prioritized replay (See paper appendix Training)
            pred_value_scalar = (
                value.detach().cpu().numpy().squeeze()
            )
            priorities[:, i] = (
                numpy.abs(pred_value_scalar - target_value_scalar[:, i])
                ** self.config.PER_alpha
            )

        loss = correctness_loss + length_loss + policy_loss
        if self.config.PER:
            # Correct PER bias by using importance-sampling (IS) weights
            loss *= weight_batch
        # Mean over batch dimension (pseudocode do a sum)
        loss = loss.mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_step += 1

        return (
            priorities,
            # For log purpose
            loss.item(),
            correctness_loss.mean().item(),
            length_loss.mean().item(),
            policy_loss.mean().item(),
        )

    def update_lr(self):
        """
        Update learning rate
        """
        lr = self.config.lr_init * self.config.lr_decay_rate ** (
            self.training_step / self.config.lr_decay_steps
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @staticmethod
    def loss_function(
        correctness_value_logits,
        length_value_logits,
        policy_logits,
        target_correctness_values,
        target_length_values,
        target_policy,
    ):
        # Cross-entropy seems to have a better convergence than MSE
        correctness_value_loss = (-target_correctness_values * torch.nn.LogSoftmax(dim=1)(correctness_value_logits)).sum(1)
        length_value_loss = (-target_length_values * torch.nn.LogSoftmax(dim=1)(length_value_logits)).sum(1)
        policy_loss = (-target_policy * torch.nn.LogSoftmax(dim=1)(policy_logits)).sum(
            1
        )
        return correctness_value_loss, length_value_loss, policy_loss
