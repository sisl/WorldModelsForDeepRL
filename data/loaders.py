import os
import torch
import numpy as np
from bisect import bisect
from os.path import join, isdir
from sklearn.model_selection import train_test_split
from envs import env_name

ROOT = f"/home/shawn/Documents/WorldModelsData/{env_name}"
# ROOT = f"/Users/shawn/Documents/WorldModelsData/{env_name}"

def get_files(root):
    folders = []
    for sd in os.listdir(root):
        samples = [join(root, sd, ssd) for ssd in os.listdir(join(root, sd)) if isdir(join(root, sd))]
        folders.append(samples)
    files = []
    isempty = False
    while not isempty:
        for i in range(len(folders)):
            if len(folders[i]) > 0:
                files.append(folders[i].pop(0))
        isempty = True
        for i in range(len(folders)):
            if len(folders[i]) > 0:
                isempty = False
    return files

class RolloutDataset(torch.utils.data.Dataset): # pylint: disable=too-few-public-methods
    def __init__(self, transform, root=ROOT, buffer_size=200, train=True): # pylint: disable=too-many-arguments
        self._transform = transform
        # self._files = [join(root, sd, ssd) for sd in os.listdir(root) if isdir(join(root, sd)) for ssd in os.listdir(join(root, sd))]
        self._files = get_files(root)
        self._files = train_test_split(self._files, train_size=0.8, shuffle=False)[1-int(train)]
        # self._files = self._files[:-400] if train else self._files[-400:]
        self._cum_size = None
        self._buffer = None
        self._buffer_fnames = None
        self._buffer_index = 0
        self._buffer_size = buffer_size

    def load_next_buffer(self):
        self._buffer_fnames = self._files[self._buffer_index:self._buffer_index + self._buffer_size]
        self._buffer_index += self._buffer_size
        self._buffer_index = self._buffer_index % len(self._files)
        self._buffer = []
        self._cum_size = [0]
        for f in self._buffer_fnames:
            with np.load(f) as data:
                self._buffer += [{k: np.copy(v) for k, v in data.items()}]
                self._cum_size += [self._cum_size[-1] + self._data_per_sequence(data['rewards'].shape[0])]

    def __len__(self):
        # to have a full sequence, you need self.seq_len + 1 elements, as
        # you must produce both an seq_len states and seq_len next_states sequences
        if not self._cum_size:
            self.load_next_buffer()
        return self._cum_size[-1]

    def __getitem__(self, i):
        # binary search through cum_size
        file_index = bisect(self._cum_size, i) - 1
        seq_index = i - self._cum_size[file_index]
        data = self._buffer[file_index]
        return self._get_data(data, seq_index)

    def _get_data(self, data, seq_index):
        pass

    def _data_per_sequence(self, data_length):
        pass

class RolloutSequenceDataset(RolloutDataset): # pylint: disable=too-few-public-methods
    """ Encapsulates rollouts.

        Rollouts should be stored in subdirs of the root directory, in the form of npz files,
        each containing a dictionary with the keys:
            - states: (rollout_len, *states_shape)
            - actions: (rollout_len, action_size)
            - rewards: (rollout_len,)
            - terminals: (rollout_len,), boolean

        As the dataset is too big to be entirely stored in rams, only chunks of it
        are stored, consisting of a constant number of files (determined by the
        buffer_size parameter).  Once built, buffers must be loaded with the
        load_next_buffer method.

        Data are then provided in the form of tuples (states, actions, next_states, reward, dones):
        - states: (seq_len, *states_shape)
        - actions: (seq_len, action_size)
        - reward: (seq_len,)
        - dones: (seq_len,) boolean
        - next_states: (seq_len, *states_shape)

        NOTE: seq_len < rollout_len in moste use cases

        :args root: root directory of data sequences
        :args seq_len: number of timesteps extracted from each rollout
        :args transform: transformation of the states
        :args train: if True, train data, else test
    """
    def __init__(self, seq_len, transform, root=ROOT, buffer_size=200, train=True): # pylint: disable=too-many-arguments
        super().__init__(transform, root, buffer_size, train)
        self._seq_len = seq_len

    def _get_data(self, data, seq_index):
        states_data = data['states'][seq_index:seq_index + self._seq_len + 1]
        states_data = self._transform(states_data.astype(np.float32))
        states, next_states = states_data[:-1], states_data[1:]
        actions = data['actions'][seq_index+1:seq_index + self._seq_len + 1].astype(np.float32)
        rewards = data['rewards'][seq_index+1:seq_index + self._seq_len + 1].astype(np.float32)
        dones = data['dones'][seq_index+1:seq_index + self._seq_len + 1].astype(np.float32)
        return states, actions, next_states, rewards, dones

    def _data_per_sequence(self, data_length):
        return data_length - self._seq_len

class RolloutObservationDataset(RolloutDataset): # pylint: disable=too-few-public-methods
    """ Encapsulates rollouts.

        Rollouts should be stored in subdirs of the root directory, in the form of npz files,
        each containing a dictionary with the keys:
            - states: (rollout_len, *states_shape)
            - actions: (rollout_len, action_size)
            - rewards: (rollout_len,)
            - terminals: (rollout_len,), boolean

        As the dataset is too big to be entirely stored in rams, only chunks of it
        are stored, consisting of a constant number of files (determined by the
        buffer_size parameter).  Once built, buffers must be loaded with the
        load_next_buffer method.

        Data are then provided in the form of images

        :args root: root directory of data sequences
        :args seq_len: number of timesteps extracted from each rollout
        :args transform: transformation of the states
        :args train: if True, train data, else test
    """
    def _data_per_sequence(self, data_length):
        return data_length

    def _get_data(self, data, seq_index):
        return self._transform(data['states'][seq_index])
