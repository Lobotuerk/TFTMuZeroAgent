"""
Enhanced AI Interface for TFT MuZero Agent Training

DEPRECATED: This module wraps ``TrainingOrchestrator`` for backward
compatibility.  New code should import from ``training_orchestrator``
directly::

    from training_orchestrator import TrainingOrchestrator, TrainingConfig

All public names are re-exported below.
"""

import asyncio
import warnings
from typing import Dict, List, Optional, Any

import config
from training_orchestrator import (
    TrainingOrchestrator,
    TrainingConfig,
    GameResult,
    create_orchestrator,
    quick_evaluation,
)

# Re-export everything so existing imports continue to work
__all__ = [
    "TrainingConfig",
    "GameResult",
    "EnhancedAIInterface",
    "AIInterface",
    "create_training_interface",
    "create_legacy_interface",
    "run_quick_evaluation",
]

warnings.warn(
    "AI_interface is deprecated – use training_orchestrator directly",
    DeprecationWarning,
    stacklevel=2,
)


class EnhancedAIInterface:
    """
    DEPRECATED wrapper around :class:`TrainingOrchestrator`.

    Kept for backward compatibility.  Prefer using
    ``TrainingOrchestrator`` directly.
    """

    _deprecated_msg = "EnhancedAIInterface is deprecated; use TrainingOrchestrator"

    def __init__(self, training_config: Optional[TrainingConfig] = None):
        self._orch = TrainingOrchestrator(training_config)

    @property
    def config(self):
        return self._orch.cfg

    @config.setter
    def config(self, value):
        self._orch.cfg = value

    @property
    def trainer(self):
        return self._orch.trainer

    @property
    def training_step(self):
        return self._orch.training_step

    @training_step.setter
    def training_step(self, value):
        self._orch.training_step = value

    @property
    def global_buffer(self):
        return self._orch.global_buffer

    @global_buffer.setter
    def global_buffer(self, value):
        self._orch.global_buffer = value

    @property
    def agent_manager(self):
        return self._orch.agent_manager

    @agent_manager.setter
    def agent_manager(self, value):
        self._orch.agent_manager = value

    @property
    def env_manager(self):
        return self._orch.env_manager

    @env_manager.setter
    def env_manager(self, value):
        self._orch.env_manager = value

    @property
    def summary_writer(self):
        return self._orch.summary_writer

    @summary_writer.setter
    def summary_writer(self, value):
        self._orch.summary_writer = value

    @property
    def training_active(self):
        return self._orch.training_active

    @training_active.setter
    def training_active(self, value):
        self._orch.training_active = value

    @property
    def games_completed(self):
        return self._orch.games_completed

    @games_completed.setter
    def games_completed(self, value):
        self._orch.games_completed = value

    @property
    def base_agent(self):
        return self._orch.base_agent

    @base_agent.setter
    def base_agent(self, value):
        self._orch.base_agent = value

    @property
    def current_weights(self):
        return self._orch.current_weights

    @current_weights.setter
    def current_weights(self, value):
        self._orch.current_weights = value

    def _setup_logging(self, run_name: str):
        return self._orch._build_logger()

    def _create_agents(self):
        # Mimic old return signature: (agent_configs, base_agent)
        self._orch.setup()
        from Models.Common_agents import CultistAgent, DivineAgent, RandomAgent
        agent_configs = [
            (self._orch._training_agents[0], 2),
            (RandomAgent("RandomTraining"), 4),
            (CultistAgent(), 1),
            (DivineAgent(), 1),
        ]
        return agent_configs, self._orch.base_agent

    async def _game_completion_callback(self, result: GameResult):
        self._orch.games_completed += 1
        if self._orch.global_buffer and self._orch.global_buffer.available_gameplay_batch():
            await self._orch._train_step()

    async def _perform_training_step(self):
        await self._orch._train_step()

    async def _run_evaluation(self):
        await self._orch.evaluate()

    async def _update_training_agents(self):
        self._orch.sync_weights()

    def _save_checkpoint(self):
        self._orch.save_checkpoint()

    async def train_torch_model(self, starting_train_step: int = 0, run_name: str = ""):
        self._orch.training_step = starting_train_step
        self._orch.cfg.starting_train_step = starting_train_step
        self._orch.cfg.run_name = run_name
        self._orch.setup()
        await self._orch.run()

    def collect_dummy_data(self):
        """Test method for simulator performance (no AI)"""
        import time
        from TFTSet4Gym.tft_set4_gym.tft_simulator import parallel_env
        asyncio.run(self._run_dummy())

    async def _run_dummy(self):
        env = parallel_env()
        while True:
            obs = env.reset()[0]
            terminated = {pid: False for pid in env.possible_agents}
            start = time.time_ns()
            while not all(terminated.values()):
                actions = {
                    a: env.action_space(a).sample()
                    for a in env.agents
                    if a in terminated and not terminated[a]
                }
                obs, _, terminated, _, _ = env.step(actions)
            print(f"Dummy game completed in {(time.time_ns() - start) / 1e9:.2f}s")

    async def run_single_evaluation(self, num_games: int = 8) -> List[GameResult]:
        return await self._orch.run_evaluation(num_games)


class AIInterface:
    """
    Legacy backward-compatibility wrapper (was the original Ray-based
    interface).  Now delegates to ``TrainingOrchestrator``.
    """

    def __init__(self):
        self._enhanced = EnhancedAIInterface()

    def train_torch_model(self, starting_train_step=0, run_name=""):
        return asyncio.run(
            self._enhanced._orch.run(max_steps=1_000_000)
        )

    def collect_dummy_data(self):
        self._enhanced.collect_dummy_data()

    def testEnv(self):
        from pettingzoo.test import parallel_api_test, api_test
        from TFTSet4Gym.tft_set4_gym.tft_simulator import env as tft_env
        print("Running PettingZoo API tests...")
        raw_env = tft_env(rank=0)
        api_test(raw_env, num_cycles=1000)
        local_env = parallel_env()
        parallel_api_test(local_env, num_cycles=1000)
        print("PettingZoo API tests completed successfully!")


# Factory functions (keep for backward compat)

def create_training_interface(config_: Optional[TrainingConfig] = None) -> EnhancedAIInterface:
    return EnhancedAIInterface(config_)


def create_legacy_interface() -> AIInterface:
    return AIInterface()


async def run_quick_evaluation(num_games: int = 8, concurrent_games: int = 2):
    return await quick_evaluation(num_games, concurrent_games)
