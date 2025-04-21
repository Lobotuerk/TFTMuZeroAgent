import ray
import torch
from Models.MCTS_torch import MCTS
from Simulator.utils import hp_from_obs, round_from_obs, t_f_c_from_obs, units_in_shop_from_obs
import config
import collections
import numpy as np
import time
import os
from Models.MuZero_torch_model import MuZeroNetwork
from Models.replay_muzero_buffer import ReplayBuffer

class MuZeroAgent:
    def __init__(self, action_size, action_limits, obs_size, simulations, global_buffer, weights=None):
        self.action_size = action_size
        self.obs_size = obs_size
        self.simulations = simulations
        self.action_limits = action_limits
        self.global_buffer = global_buffer
        self.shared_weights = weights
        self.model = MuZeroNetwork()
        self.mcts = MCTS(sample_size=80, action_size=self.action_size, action_limits=self.action_limits, policy_size=1000, network=self.model)
        if weights is not None:
            self.model.load_state_dict(weights)
        self.model.to('cuda')
        self.hp = []
        self.replay_buffers = []

    def select_action(self, observation, mask, reward, terminated):
        while len(self.replay_buffers) < observation.shape[0]:
            self.replay_buffers.append(ReplayBuffer(self.global_buffer))
            self.hp.append(100)
        action, policy = self.mcts.generate_action(self.simulations, observation=observation, mask=mask)
        if np.any(terminated):
            for n in range(len(terminated)):
                if terminated[n]:
                    self.replay_buffers[n].store_step(observation[n], action[n], reward[n], policy[n])
                    self.replay_buffers[n].move_buffer_to_global()
                    # print(f'Muzero {n} ended with reward {reward[n]}')
        for n in range(len(observation)):
            if not terminated[n]:
                self.replay_buffers[n].store_step(observation[n], action[n], reward[n], policy[n])
            turns_left = t_f_c_from_obs(observation[n])
            round = round_from_obs(observation[n])
            if turns_left == config.ACTIONS_PER_TURN and round > 1:
                hp = hp_from_obs(observation[n])
                self.global_buffer.store_combat.remote([observation[n], 1 if hp >= self.hp[n] else -1])
                self.hp[n] = hp
        actions = [[int(x) for x in a.split("_")] for a in action]
        for i, model_action in enumerate(actions):
            action[i] = self.translate_action(model_action, units_in_shop_from_obs(observation[i]))
        # action = np.random.randint(self.action_limits, size=(observation.shape[0], self.action_size))
        # print(action)
        return action
    
    def get_weights(self):
        return self.model.get_weights()
    
    def translate_action(self, action, units_in_shop):
        ret_action = action
        if action[0] == 1:
            if units_in_shop[0] is not ' ':
                ret_action = [2, 0, 0, 0]
            else:
                ret_action = [4, 0, 0, 0]
        elif action[0] == 2:
            ret_action = [5, 0, 0, 0]
        return ret_action
    
class BaseMuZeroAgent(MuZeroAgent):
    pass