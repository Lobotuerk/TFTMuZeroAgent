import ray
import config
import datetime
import numpy as np
from collections import deque
import os
import random
import pickle


@ray.remote
class GlobalBuffer:
    def __init__(self, batch_size):
        self.gameplay_experiences = deque(maxlen=500)
        self.combat_experiences = deque(maxlen=500)
        self.batch_size = batch_size
        self.current_gameplay_index = None
        self.current_combat_index = None
        self.current_gameplay_order = None
        self.current_combat_order = None

    # Might be a bug with the action_batch not always having correct dims
    def sample_gameplay_batch(self, batch_size):
        # Returns: a batch of gameplay experiences without regard to which agent.
        observation_batch = []
        action_batch = []
        value_batch = []
        reward_batch = []
        policy_batch = []
        for _ in range(batch_size):
            observation, action, value, reward, policy = self.gameplay_experiences.pop()
            observation_batch.append(observation)
            action_batch.append(action)
            value_batch.append(value)
            reward_batch.append(reward)
            policy_batch.append(policy)

        observation_batch = np.array(observation_batch)
        action_batch = np.array(action_batch)
        value_batch = np.array(value_batch)
        reward_batch = np.array(reward_batch)
        policy_batch = np.array(policy_batch)
        # print("observation ", observation_batch.shape)
        # print("action ", action_batch.shape)
        # print("value ", value_batch.shape)
        # print("reward ", reward_batch.shape)
        # print("policy ", policy_batch.shape)
        # policy_mask_batch = np.asarray(policy_mask_batch).astype('float32')

        return [observation_batch, action_batch, value_batch, reward_batch, policy_batch]
    
    def sample_combat_batch(self, batch_size):
        # Returns: a batch of gameplay experiences without regard to which agent.
        observation_batch = []
        result_batch = []
        for _ in range(batch_size):
            observation, result = self.combat_experiences.pop()
            observation_batch.append(observation)
            result_batch.append(result)

        observation_batch = np.array(observation_batch)
        result_batch = np.array(result_batch)

        return [observation_batch, result_batch]

    def store_episode(self, sample):
        self.gameplay_experiences.extend(sample)
        while len(self.gameplay_experiences) > self.batch_size:
            data = self.sample_gameplay_batch(self.batch_size)
            # Count batchs in the buffer folder
            if not os.path.exists(config.GAMEPLAY_BUFFER_PATH):
                os.makedirs(config.GAMEPLAY_BUFFER_PATH)
            # grab the current time up to the second
            time_stamp = datetime.datetime.now().strftime("%H%M%S")
            batch_file = os.path.join(config.GAMEPLAY_BUFFER_PATH, f"batch_{time_stamp}.pickle")
            # Save batch to file
            with open(batch_file, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def store_combat(self, sample):
        self.combat_experiences.append(sample)
        while len(self.combat_experiences) > self.batch_size:
            data = self.sample_combat_batch(self.batch_size)
            # Count batchs in the buffer folder
            if not os.path.exists(config.COMBAT_BUFFER_PATH):
                os.makedirs(config.COMBAT_BUFFER_PATH)
            time_stamp = datetime.datetime.now().strftime("%H%M%S")
            batch_file = os.path.join(config.COMBAT_BUFFER_PATH, f"batch_{time_stamp}.pickle")
            with open(batch_file, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def available_gameplay_batch(self):
        # queue_length = len(self.gameplay_experiences)
        files_lenght = len(os.listdir(config.GAMEPLAY_BUFFER_PATH))
        # print("QUEUE SIZE: ", queue_length)
        if files_lenght >= 1:
            return True
        return False
    
    def available_combat_batch(self):
        # queue_length = len(self.combat_experiences)
        files_lenght = len(os.listdir(config.COMBAT_BUFFER_PATH))
        if files_lenght >= 1:
            return True
        return False
    
    def read_gameplay_batch(self):
        # Read batch from file
        if not os.path.exists(config.GAMEPLAY_BUFFER_PATH):
            os.makedirs(config.GAMEPLAY_BUFFER_PATH)
        if self.current_gameplay_order is None:
            files = os.listdir(config.GAMEPLAY_BUFFER_PATH)
            random.shuffle(files)
            self.current_gameplay_order = files
            self.current_gameplay_index = 0
        if self.current_gameplay_index >= len(self.current_gameplay_order):
            self.current_gameplay_index = 0
        batch_file = os.path.join(config.GAMEPLAY_BUFFER_PATH, self.current_gameplay_order[self.current_gameplay_index])
        with open(batch_file, 'rb') as handle:
                data = pickle.load(handle)
                self.current_gameplay_index += 1
        return data
    
    def read_combat_batch(self):
        # Read batch from file
        if not os.path.exists(config.COMBAT_BUFFER_PATH):
            os.makedirs(config.COMBAT_BUFFER_PATH)
        if self.current_combat_order is None:
            files = os.listdir(config.COMBAT_BUFFER_PATH)
            random.shuffle(files)
            self.current_combat_order = files
            self.current_combat_index = 0
        if self.current_combat_index >= len(self.current_combat_order):
            self.current_combat_index = 0
        batch_file = os.path.join(config.COMBAT_BUFFER_PATH, self.current_combat_order[self.current_combat_index])
        with open(batch_file, 'rb') as handle:
                data = pickle.load(handle)
                self.current_combat_index += 1
        return data
    
    def clear_gameplay_buffer(self):
        # Clear the buffer
        if os.path.exists(config.GAMEPLAY_BUFFER_PATH):
            for file in os.listdir(config.GAMEPLAY_BUFFER_PATH):
                os.remove(os.path.join(config.GAMEPLAY_BUFFER_PATH, file))
        self.current_gameplay_index = None
        self.current_gameplay_order = None

    def clear_combat_buffer(self):
        # Clear the buffer
        if os.path.exists(config.COMBAT_BUFFER_PATH):
            for file in os.listdir(config.COMBAT_BUFFER_PATH):
                os.remove(os.path.join(config.COMBAT_BUFFER_PATH, file))
        self.current_combat_index = None
        self.current_combat_order = None

    def get_gameplay_buffer_size(self):
        # Get the size of the buffer
        self.current_gameplay_index = 0
        if os.path.exists(config.GAMEPLAY_BUFFER_PATH):
            return len(os.listdir(config.GAMEPLAY_BUFFER_PATH))
        return 0

    def get_combat_buffer_size(self):
        # Get the size of the buffer
        self.current_combat_index = 0
        if os.path.exists(config.COMBAT_BUFFER_PATH):
            return len(os.listdir(config.COMBAT_BUFFER_PATH))
        return 0
