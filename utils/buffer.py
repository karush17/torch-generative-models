"""Implements the data replay buffer."""

from typing import Any, List, Dict

import abc
import numpy as np

class ReplayBuffer(object, metaclass=abc.ABCMeta):
    """Implements the data replay buffer."""
    @abc.abstractmethod
    def add_sample(self, observation: np.ndarray, action: np.ndarray,
                   reward: np.ndarray, next_observation: np.ndarray,
                   terminal: np.ndarray, **kwargs):
        """Adds a sample to the buffer."""
        pass

    @abc.abstractmethod
    def terminate_episode(self):
        """Terminates the episode."""
        pass

    @abc.abstractmethod
    def num_steps_can_sample(self, **kwargs):
        """Returns the number of steps that can be sampled."""
        pass

    def add_path(self, path: Dict[str, Any]):
        """Adds a path to the buffer."""
        for i, (
                obs,
                action,
                reward,
                next_obs,
                terminal,
                agent_info,
                env_info
        ) in enumerate(zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
            path["agent_infos"],
            path["env_infos"],
        )):
            self.add_sample(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                terminal=terminal,
                agent_info=agent_info,
                env_info=env_info,
            )
        self.terminate_episode()

    def add_paths(self, paths: List[Dict[str, Any]]):
        """Adds multiple paths to the buffer."""
        for path in paths:
            self.add_path(path)

    @abc.abstractmethod
    def random_batch(self, batch_size: int):
        """Samples a random batch."""
        pass

    def get_diagnostics(self):
        """Returns logging diagnostics."""
        return {}

    def get_snapshot(self):
        """Gets a snapshot of the buffer."""
        return {}

    def end_epoch(self, epoch: int):
        """Ends the current iteration."""
        return
