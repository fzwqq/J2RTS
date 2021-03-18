import random
from dataclasses import dataclass
from typing import List, Any
from microrts.rts_wrapper.envs.units_name import *
from microrts.rts_wrapper.envs.utils_jin import unit_feature_encoder_v2
from microrts.rts_wrapper.envs.utils import rd, utt_feature
from dacite import from_dict
import numpy as np
import torch


@dataclass
class Transition:
    obs_t: np.array
    # action  : List[Any] # list of (Unit, int(network_action) )
    action: Any
    obs_tp1: np.array
    # value   : float
    # pi_sa   : float
    reward: float
    hxs: np.array
    done: bool
    duration: float
    ctf: np.array
    irew: np.array


@dataclass
class Batches:
    states: np.array
    units: np.array
    utts: np.array
    ua_type: np.array
    ua_parm: np.array
    ua_prod: np.array
    next_states: np.array
    rewards: np.array
    hxses: np.array
    done: np.array
    durations: np.array
    ctfs: np.array
    irews: np.array

    def to(self, device):
        done_masks = torch.FloatTensor(
            [0.0 if _done == 1 else 1.0 for _done in self.done]
        )
        # print(done_masks)
        return torch.from_numpy(self.states).float().to(device), \
               torch.from_numpy(self.units).float().to(device), \
               torch.from_numpy(self.utts).float().to(device), \
               torch.from_numpy(self.ua_type).long().to(device).unsqueeze(1), \
               torch.from_numpy(self.ua_parm).long().to(device).unsqueeze(1), \
               torch.from_numpy(self.ua_prod).long().to(device).unsqueeze(1), \
               torch.from_numpy(self.next_states).float().to(device), \
               torch.from_numpy(self.rewards).float().to(device).unsqueeze(1), \
               torch.from_numpy(self.hxses).float().to(device) if self.hxses.all() else self.hxses, \
               done_masks.to(device).unsqueeze(1), \
               torch.from_numpy(self.durations).float().to(device).unsqueeze(1), \
               torch.from_numpy(self.ctfs).float().to(device).unsqueeze(1), \
               torch.from_numpy(self.irews).float().to(device).unsqueeze(1), \
            # torch.from_numpy(self.done).int().to(device).unsqueeze(1)


class ReplayBuffer2(object):
    def __init__(self, size, frame_history_len=1):
        """Create Replay buffer
        Arguments:
            size {int} -- Storage capacity i.e. xax number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.

        Keyword Arguments:
            frame_history_len {int} -- Num of frames taken for training input (default: {1})
        """

        self._storage = []
        self._maxsize = size
        self._next_idx = 0  # next pos to store the new data
        self._frame_history_len = frame_history_len

    def __len__(self):
        """Show current amount of samples

        Returns:
            int -- how many samples stored in the buffer?
        """
        return len(self._storage)

    def refresh(self):
        self._storage.clear()
        self._next_idx = 0

    def shuffle(self):
        random.shuffle(self._storage)

    def push(self, **kwargs):
        """Saves a transition

        Arguments:
            obs_t {np.array} -- [description]
            action {Unit and action pair} -- [description]
            reward {float} -- [description]
            obs_tp1 {np.array} -- [description]
            done {bool} -- [description]
            hxs {np.array} -- [description]
        """
        trans = Transition(**kwargs)

        if self._next_idx >= len(self._storage):
            self._storage.append(trans)
        else:
            self._storage[self._next_idx] = trans
            # self._storage[self._last_idx]["obs_tp1"] = trans["obs_tp1"]
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        """[summary]

        Arguments:
            idxes {[type]} -- [description]

        Returns: Batch list of
            states {np.array} --
            units {np.array} -- unit features
            utts {np.array} -- utt features
            actions {np.array} -- network action for states
            next_states {np.array} --
            rewards {np.array} --
            done_masks {np.array}

        """
        states, units, utts, ua_type, ua_parm, ua_prod, next_states, rewards, hxses, done_masks = [], [], [], [], [], [], [], [], [], []
        durations = []
        ctfs = []
        irews = []
        for i in idxes:
            transition = self._storage[i]
            state, unit_action, next_state, reward,hxs, done, duration, ctf, irew = transition.__dict__.values()
            map_size = state.shape[-2:]

            u, _type, _parm, _prod = unit_action
            states.append(state)
            units.append(unit_feature_encoder_v2(u,map_size))
            utts.append(utt_feature),
            ua_type.append(_type),
            ua_parm.append(_parm),
            ua_prod.append(_prod),
            next_states.append(next_state)
            hxses.append(hxs)
            rewards.append(reward)
            done_masks.append(done)
            durations.append(duration)
            ctfs.append(ctf)
            irews.append(irew)
        encoded_samples = {
            "states":  np.array(states),
            "units":   np.array(units),
            "utts":    np.array(utts),
            "ua_type": np.array(ua_type),
            "ua_parm": np.array(ua_parm),
            "ua_prod": np.array(ua_prod),
            "next_states": np.array(next_states),
            "rewards": np.array(rewards),
            "hxses":   np.array(hxses),
            "done":    np.array(done_masks),
            "durations": np.array(durations),
            "ctfs":     np.array(ctfs),
            "irews":    np.array(irews),
        }
        return encoded_samples

    def sample(self, batch_size):
        if batch_size == "all":
            batch_size = self.__len__()
            idxes = [i for i in range(batch_size)]
        else:
            idxes = [rd.randint(0, len(self._storage)) for _ in range(batch_size)]
        encoded_samples = self._encode_sample(idxes)
        return Batches(**encoded_samples)

    def fix_last_mask(self, done):
        # if has len >= 1
        if self._next_idx:
            self._storage[-1].done = done