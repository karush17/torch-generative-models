import numpy as np
import typing
from gym.spaces import Dict, Discrete
from utils.buffer import ReplayBuffer

class ObsDictReplayBuffer(ReplayBuffer):
    def __init__(
            self,
            max_size,
            env,
            ob_keys_to_save=None,
            internal_keys=None,
            observation_key='observation',
            save_data_in_snapshot=False,
            biased_sampling=False,
            bias_point=None,
            before_bias_point_probability=0.5,
    ):
        if ob_keys_to_save is None:
            ob_keys_to_save = []
        else:
            ob_keys_to_save = list(ob_keys_to_save)
        if internal_keys is None:
            internal_keys = [observation_key]
        self.internal_keys = internal_keys
        assert isinstance(env.observation_space, Dict)
        self.max_size = max_size
        self.env = env
        self.ob_keys_to_save = ob_keys_to_save
        self.observation_key = observation_key
        self.save_data_in_snapshot = save_data_in_snapshot

        self.biased_sampling = biased_sampling
        self.bias_point = bias_point
        self.before_bias_point_probability = before_bias_point_probability

        self._action_dim = env.action_space.low.size
        self._actions = np.zeros((max_size, self._action_dim), dtype=np.float32)
        self._terminals = np.zeros((max_size, 1), dtype='uint8')
        self._rewards = np.zeros((max_size, 1))
        self._obs = {}
        self._next_obs = {}
        self.ob_spaces = self.env.observation_space.spaces

        for key in self.ob_keys_to_save + internal_keys:
            assert key in self.ob_spaces, \
                "Key not found in the observation space: %s" % key
            type = np.float64
            if key.startswith('image'):
                type = np.uint8
            self._obs[key] = np.zeros(
                (max_size, self.ob_spaces[key].low.size), dtype=type)
            self._next_obs[key] = np.zeros(
                (max_size, self.ob_spaces[key].low.size), dtype=type)
        self._top = 0
        self._size = 0
        self._idx_to_future_obs_idx = [None] * max_size

        if isinstance(self.env.action_space, Discrete):
            raise NotImplementedError("TODO")

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        raise NotImplementedError("Only use add_path")

    def terminate_episode(self):
        pass

    def num_steps_can_sample(self):
        return self._size

    def add_paths_from_mdp(self, paths):
        for path in paths:
            self.add_path_from_mdp(path)

    def add_path_from_mdp(self, path):
        obs = path["observations"]
        actions = path["actions"]
        rewards = path["rewards"]
        next_obs = path["next_observations"]
        terminals = path["terminals"]
        path_len = len(rewards)
        obs = [obs_[0] for obs_ in obs]
        next_obs = [next_obs_[0] for next_obs_ in next_obs]
        obs = flatten_dict(list(obs), self.ob_keys_to_save + self.internal_keys)
        next_obs = flatten_dict(list(next_obs),
                                self.ob_keys_to_save + self.internal_keys)
        obs = preprocess_obs_dict(obs)
        next_obs = preprocess_obs_dict(next_obs)
        self.add_processed_path(path_len, actions, terminals,
                                obs, next_obs, rewards)

    def add_path(self, path):
        obs = path["observations"]
        actions = path["actions"]
        rewards = path["rewards"]
        next_obs = path["next_observations"]
        terminals = path["terminals"]
        path_len = len(rewards)
        actions = flatten_n(actions)
        obs = flatten_dict(obs, self.ob_keys_to_save + self.internal_keys)
        next_obs = flatten_dict(next_obs,
                                self.ob_keys_to_save + self.internal_keys)
        obs = preprocess_obs_dict(obs)
        next_obs = preprocess_obs_dict(next_obs)
        self.add_processed_path(path_len, actions, terminals,
                                obs, next_obs, rewards)

    def add_processed_path(self, path_len, actions, terminals,
                           obs, next_obs, rewards):
        if self._top + path_len >= self.max_size:
            num_pre_wrap_steps = self.max_size - self._top
            pre_wrap_buffer_slice = np.s_[
                                    self._top:self._top + num_pre_wrap_steps, :
                                    ]
            pre_wrap_path_slice = np.s_[0:num_pre_wrap_steps, :]
            num_post_wrap_steps = path_len - num_pre_wrap_steps
            post_wrap_buffer_slice = slice(0, num_post_wrap_steps)
            post_wrap_path_slice = slice(num_pre_wrap_steps, path_len)
            for buffer_slice, path_slice in [
                (pre_wrap_buffer_slice, pre_wrap_path_slice),
                (post_wrap_buffer_slice, post_wrap_path_slice),
            ]:
                self._actions[buffer_slice] = actions[path_slice]
                self._terminals[buffer_slice] = terminals[path_slice]
                self._rewards[buffer_slice] = terminals[path_slice]
                for key in self.ob_keys_to_save + self.internal_keys:
                    self._obs[key][buffer_slice] = obs[key][path_slice]
                    self._next_obs[key][buffer_slice] = next_obs[key][
                        path_slice]
            for i in range(self._top, self.max_size):
                self._idx_to_future_obs_idx[i] = np.hstack((
                    np.arange(i, self.max_size),
                    np.arange(0, num_post_wrap_steps)
                ))
            for i in range(0, num_post_wrap_steps):
                self._idx_to_future_obs_idx[i] = np.arange(
                    i,
                    num_post_wrap_steps,
                )
        else:
            slc = np.s_[self._top:self._top + path_len, :]
            self._actions[slc] = actions
            self._terminals[slc] = terminals
            self._rewards[slc] = rewards
            for key in self.ob_keys_to_save + self.internal_keys:
                self._obs[key][slc] = obs[key]
                self._next_obs[key][slc] = next_obs[key]
            for i in range(self._top, self._top + path_len):
                self._idx_to_future_obs_idx[i] = np.arange(
                    i, self._top + path_len
                )
        self._top = (self._top + path_len) % self.max_size
        self._size = min(self._size + path_len, self.max_size)

    def _sample_indices(self, batch_size):
        if self.biased_sampling:
            assert self.bias_point is not None
            indices_1 = np.random.randint(0, self.bias_point, batch_size)
            indices_2 = np.random.randint(self.bias_point, self._size, batch_size)
            biased_coin_flip = (np.random.uniform(size=batch_size) <
                                self.before_bias_point_probability) * 1
            indices = np.where(biased_coin_flip, indices_1, indices_2)
        else:
            indices = np.random.randint(0, self._size, batch_size)
        return indices

    def random_batch(self, batch_size):

        indices = self._sample_indices(batch_size)
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        terminals = self._terminals[indices]
        obs = self._obs[self.observation_key][indices]
        next_obs = self._next_obs[self.observation_key][indices]
        if self.observation_key == 'image':
            obs = normalize_image(obs)
            next_obs = normalize_image(next_obs)
        batch = {}
        batch.update({
            'observations': obs,
            'actions': actions,
            'rewards': rewards,
            'terminals': terminals,
            'next_observations': next_obs,
            'indices': np.array(indices).reshape(-1, 1),
        })
        return batch

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        if self.save_data_in_snapshot:
            snapshot.update({
                'observations': self.get_slice(self._obs, slice(0, self._top)),
                'next_observations': self.get_slice(
                    self._next_obs, slice(0, self._top)
                ),
                'actions': self._actions[:self._top],
                'terminals': self._terminals[:self._top],
                'rewards': self._rewards[:self._top],
                'idx_to_future_obs_idx': (
                    self._idx_to_future_obs_idx[:self._top]
                ),
            })
        return snapshot

    def get_slice(self, obs_dict, slc):
        new_dict = {}
        for key in self.ob_keys_to_save + self.internal_keys:
            new_dict[key] = obs_dict[key][slc]
        return new_dict

    def get_diagnostics(self):
        return {'top': self._top}


def flatten_n(xs):
    xs = np.asarray(xs)
    return xs.reshape((xs.shape[0], -1))

def flatten_dict(dicts, keys):
    return {
        key: flatten_n([d[key] for d in dicts])
        for key in keys
    }

def preprocess_obs_dict(obs_dict):
    for obs_key, obs in obs_dict.items():
        if 'image' in obs_key and obs is not None:
            obs_dict[obs_key] = unnormalize_image(obs)
    return obs_dict

def postprocess_obs_dict(obs_dict):
    for obs_key, obs in obs_dict.items():
        if 'image' in obs_key and obs is not None:
            obs_dict[obs_key] = normalize_image(obs)
    return obs_dict

def normalize_image(image):
    assert image.dtype == np.uint8
    return np.float64(image) / 255.0

def unnormalize_image(image):
    assert image.dtype != np.uint8
    return np.uint8(image * 255.0)
