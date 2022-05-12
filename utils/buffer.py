import abc

class ReplayBuffer(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def add_sample(self, observation, action, reward, next_observation,
                   terminal, **kwargs):
        pass

    @abc.abstractmethod
    def terminate_episode(self):
        pass

    @abc.abstractmethod
    def num_steps_can_sample(self, **kwargs):
        pass

    def add_path(self, path):
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

    def add_paths(self, paths):
        for path in paths:
            self.add_path(path)

    @abc.abstractmethod
    def random_batch(self, batch_size):
        pass

    def get_diagnostics(self):
        return {}

    def get_snapshot(self):
        return {}

    def end_epoch(self, epoch):
        return