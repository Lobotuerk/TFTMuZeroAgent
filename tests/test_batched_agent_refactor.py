import unittest
import numpy as np
import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Models.Common_agents import BaseAgent, RandomAgent
from Models.global_buffer import GlobalBuffer

class MockBatchedAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_impl_called = 0

    def _select_action_impl(self, obs, mask, reward=None, terminated=None, precomputed_results=None):
        return [0, 0, 0]

    def _batch_select_action_impl(self, observations, masks, rewards=None, terminated=None, precomputed_results=None, **kwargs):
        self.batch_impl_called += 1
        return super()._batch_select_action_impl(observations, masks, rewards, terminated, precomputed_results, **kwargs)

class TestBatchedAgentRefactor(unittest.TestCase):
    def test_batch_select_action_calls_impl(self):
        agent = MockBatchedAgent()
        obs = [np.zeros(5152) for _ in range(4)]
        masks = [None] * 4
        
        actions = agent.batch_select_action(obs, masks)
        
        self.assertEqual(len(actions), 4)
        self.assertEqual(agent.batch_impl_called, 1)

    def test_combat_tracking_in_batch(self):
        gb = GlobalBuffer()
        agent = RandomAgent(global_buffer=gb)
        
        def create_obs(turns, hp):
            return {
                'tensor': np.zeros(5152),
                'turns_for_combat': turns,
                'health': hp
            }

        # Step 1: Initialize states
        agent.batch_select_action(
            [create_obs(0, 100), create_obs(0, 100)],
            [None, None],
            player_ids=["p1", "p2"]
        )
        
        self.assertEqual(agent.prev_turns_for_combat["p1"], 0)
        self.assertEqual(agent.prev_turns_for_combat["p2"], 0)
        
        # Step 2: Combat ends for both
        # p1 wins (100 -> 100), p2 loses (100 -> 80)
        agent.batch_select_action(
            [create_obs(10, 100), create_obs(10, 80)],
            [None, None],
            player_ids=["p1", "p2"]
        )
        
        self.assertEqual(len(gb.combat_experiences), 2)
        
        # Check results
        results = [exp[1] for exp in gb.combat_experiences]
        self.assertIn(1.0, results)
        self.assertIn(0.0, results)

    def test_experience_storage_in_batch(self):
        gb = GlobalBuffer()
        # Mock global_buffer might not have everything, let's use the real one or check it
        agent = RandomAgent(global_buffer=gb, save_data=True)
        
        obs = [np.zeros(5152) for _ in range(3)]
        agent.batch_select_action(obs, [None]*3, player_ids=["p1", "p2", "p3"])
        
        # Check that steps were stored in player buffers
        self.assertEqual(len(agent.replay_buffers["p1"].observations), 1)
        self.assertEqual(len(agent.replay_buffers["p2"].observations), 1)
        self.assertEqual(len(agent.replay_buffers["p3"].observations), 1)

if __name__ == '__main__':
    unittest.main()
