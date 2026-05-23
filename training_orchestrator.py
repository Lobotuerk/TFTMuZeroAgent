"""
Training Orchestrator for TFT MuZero Agent

Drives the RL lifecycle: Collect -> Train -> Sync -> Evaluate

Explicit lifecycle phases:
  1. COLLECT: Run games in parallel via ParallelEnvironmentManager,
     gather experience into the GlobalBuffer
  2. TRAIN:   Sample from the buffer, update the model via Trainer
  3. SYNC:    Distribute the updated weights to the active collection
     agents so they immediately benefit from the new policy
  4. EVALUATE:Periodically pit the new model against the old one;
     keep the best performing weights
"""

import asyncio
import time
import copy
import datetime
import os
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from collections import defaultdict
import torch
from torch.utils.tensorboard import SummaryWriter

import config
from Models.global_buffer import GlobalBuffer
from Models.MuZero_torch_trainer import Trainer
from Models.MuZero_torch_agent import MuZeroAgent
from Models.Common_agents import CultistAgent, DivineAgent, RandomAgent
from Models.enhanced_agent_interface import (
    create_enhanced_setup,
    create_custom_agent_setup,
    AsyncGameEnvironment,
    BatchInferenceServer,
    EnhancedAgentManager,
)
from TFTSet4Gym.tft_set4_gym.tft_simulator import parallel_env


# ---------------------------------------------------------------------------
# Data objects
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    starting_train_step: int = 0
    run_name: str = ""
    save_interval: int = config.CHECKPOINT_STEPS
    evaluation_interval: int = config.CHECKPOINT_STEPS
    concurrent_games: int = config.CONCURRENT_GAMES
    evaluation_games: int = config.EVALUATION_GAMES
    evaluation_concurrent: int = config.EVALUATION_CONCURRENT_GAMES
    max_batch_size: int = 16
    batch_timeout_ms: float = 5.0
    gpu_memory_fraction: float = 0.7


@dataclass
class GameResult:
    """Container for a single game outcome"""
    game_id: str
    placements: Dict[str, int]
    scores: Dict[str, float]
    duration: float
    agent_mapping: Dict[str, type]


# ---------------------------------------------------------------------------
# Game worker – runs one async game
# ---------------------------------------------------------------------------

class _GameWorker:
    """
    Async game worker that replaces the legacy Ray DataWorker.
    Runs a single game asynchronously without Ray overhead.
    """

    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.games_completed = 0

    async def run_game(self,
                       agent_manager: EnhancedAgentManager,
                       return_placements: bool = False) -> GameResult:
        """Run a single game end-to-end and return the result."""
        try:
            game_id = f"worker_{self.worker_id}_game_{self.games_completed}"
            start_time = time.time()

            env = parallel_env(rank=self.worker_id)
            observations = env.reset()[0]
            terminated = {pid: False for pid in env.possible_agents}
            rewards = {pid: 0.0 for pid in env.possible_agents}
            scores = {pid: 0.0 for pid in env.possible_agents}

            step_count = 0
            while not all(terminated.values()):
                step_count += 1
                float_rewards = {k: float(v) for k, v in rewards.items()}

                try:
                    actions_task = agent_manager.get_actions(
                        observations, float_rewards, terminated
                    )
                    actions = await asyncio.wait_for(actions_task, timeout=2.0)
                except (asyncio.TimeoutError, Exception):
                    actions = {}
                    for pid in observations:
                        if not terminated.get(pid, True):
                            actions[pid] = [
                                random.randint(0, 5),
                                random.randint(0, 36),
                                random.randint(0, 27),
                            ]
                        else:
                            actions[pid] = [0, 0, 0]

                processed = {}
                for pid, action in actions.items():
                    if terminated.get(pid, True):
                        processed[pid] = [0, 0, 0]
                        continue
                    if isinstance(action, (list, np.ndarray)) and len(action) >= 3:
                        processed[pid] = action[:3]
                    elif hasattr(action, "tolist"):
                        lst = action.tolist()
                        processed[pid] = lst[:3] if isinstance(lst, list) and len(lst) >= 3 else [0, 0, 0]
                    else:
                        processed[pid] = [0, 0, 0]
                actions = processed

                observations, rewards, terminated, _, _ = env.step(actions)
                for p in terminated:
                    if terminated[p]:
                        scores[p] = rewards[p]

                if step_count > 1000:
                    break

            placements = {}
            if return_placements:
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                for i, (pid, _) in enumerate(sorted_scores):
                    placements[pid] = i + 1

            agent_mapping = agent_manager.get_player_agent_mapping() if return_placements else {}
            duration = time.time() - start_time
            self.games_completed += 1

            return GameResult(
                game_id=game_id,
                placements=placements,
                scores=scores,
                duration=duration,
                agent_mapping=agent_mapping,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise


# ---------------------------------------------------------------------------
# Parallel environment manager
# ---------------------------------------------------------------------------

class _ParallelEnvManager:
    """
    Manages N concurrent game workers without Ray.
    Supports continuous execution (auto-restart) and fixed-run evaluation.
    """

    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        self.workers = [_GameWorker(i) for i in range(num_workers)]
        self.active_tasks: Dict[asyncio.Task, int] = {}
        self.should_continue = True

    def stop(self):
        self.should_continue = False

    async def run_continuously(self,
                               agent_manager: EnhancedAgentManager,
                               on_game_done: Optional[Callable] = None) -> None:
        """Run games back-to-back, starting a new one as soon as one finishes."""
        self.active_tasks.clear()
        for i, worker in enumerate(self.workers):
            task = asyncio.create_task(worker.run_game(agent_manager))
            self.active_tasks[task] = i

        games_completed = 0
        while self.should_continue:
            if not self.active_tasks:
                for i, w in enumerate(self.workers):
                    task = asyncio.create_task(w.run_game(agent_manager))
                    self.active_tasks[task] = i
                continue

            try:
                done, _ = await asyncio.wait(
                    self.active_tasks.keys(),
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                continue

            for completed in done:
                worker_id = self.active_tasks.pop(completed, -1)
                self._discard(completed)
                try:
                    result = await completed
                    games_completed += 1
                    if on_game_done:
                        await on_game_done(result)
                except Exception as e:
                    pass

                if worker_id >= 0:
                    new = asyncio.create_task(self.workers[worker_id].run_game(agent_manager))
                    self.active_tasks[new] = worker_id

        for t in list(self.active_tasks):
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self.active_tasks.clear()

    async def run_fixed_games(self,
                              agent_manager: EnhancedAgentManager,
                              num_games: int) -> List[GameResult]:
        """Run exactly *num_games* evaluation games and return their results."""
        games_per_worker = num_games // self.num_workers
        remainder = num_games % self.num_workers
        tasks = []
        for i, worker in enumerate(self.workers):
            count = games_per_worker + (1 if i < remainder else 0)
            for _ in range(count):
                tasks.append(asyncio.create_task(worker.run_game(agent_manager, return_placements=True)))

        results = []
        if tasks:
            completed = await asyncio.gather(*tasks, return_exceptions=True)
            for r in completed:
                if isinstance(r, GameResult):
                    results.append(r)
        return results

    def _discard(self, task):
        self.active_tasks.pop(task, None)


# ---------------------------------------------------------------------------
# TrainingOrchestrator
# ---------------------------------------------------------------------------

class TrainingOrchestrator:
    """
    Central orchestrator that drives the RL lifecycle:

        Collect -> Train -> Sync -> Evaluate

    Usage::

        orch = TrainingOrchestrator(config)
        orch.setup()
        await orch.run(max_steps=100_000)

    For single-episode debugging::

        result = await orch.run_single_episode()

    For a parallel demo (no training) see :meth:`run_parallel_demo`.
    """

    def __init__(self, training_config: Optional[TrainingConfig] = None):
        self.cfg = training_config or TrainingConfig()

        # Components (created in setup)
        self.trainer: Optional[Trainer] = None
        self.global_buffer: Optional[GlobalBuffer] = None
        self.agent_manager: Optional[EnhancedAgentManager] = None
        self.env_manager: Optional[_ParallelEnvManager] = None
        self.summary_writer: Optional[SummaryWriter] = None

        # Training state
        self.training_step: int = self.cfg.starting_train_step
        self.games_completed: int = 0
        self.training_active: bool = False

        # Model / weights
        self.base_agent: Optional[MuZeroAgent] = None
        self.current_weights: Optional[Dict] = None
        self._training_agents: List[MuZeroAgent] = []

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self):
        """Create all components: buffer, agents, batch processor, trainer."""
        self.trainer = Trainer()
        self.summary_writer = self._build_logger()
        self.global_buffer = GlobalBuffer(config.BATCH_SIZE)

        # --- agent config -------------------------------------------------
        self.base_agent = MuZeroAgent(
            action_size=3,
            action_limits=[7, 37, 10],
            obs_size=config.OBSERVATION_SIZE,
            simulations=config.NUM_SIMULATIONS,
            global_buffer=self.global_buffer,
        )

        if self.training_step > 0:
            ckpt = f"./checkpoint/checkpoint_{self.training_step}"
            if os.path.isfile(ckpt):
                self.base_agent.model.load_state_dict(torch.load(ckpt))

        self.current_weights = copy.deepcopy(self.base_agent.get_weights())

        # MuZero agents for *collection* – start with current weights
        training_muzero = MuZeroAgent(
            action_size=3,
            action_limits=[7, 37, 10],
            obs_size=config.OBSERVATION_SIZE,
            simulations=config.NUM_SIMULATIONS,
            global_buffer=self.global_buffer,
            weights=copy.deepcopy(self.current_weights),
        )
        self._training_agents = [training_muzero]

        random_agent = RandomAgent("RandomTraining")
        cultist_agent = CultistAgent()
        divine_agent = DivineAgent()

        agent_configs: List[Tuple[Any, int]] = [
            (training_muzero, 2),
            (random_agent, 4),
            (cultist_agent, 1),
            (divine_agent, 1),
        ]

        # --- batch processor + agent manager -------------------------------
        self.agent_manager, _ = create_custom_agent_setup(
            agent_configs,
            max_batch_size=self.cfg.max_batch_size,
            batch_timeout_ms=self.cfg.batch_timeout_ms,
            gpu_memory_fraction=self.cfg.gpu_memory_fraction,
        )

        # --- parallel env manager -----------------------------------------
        self.env_manager = _ParallelEnvManager(self.cfg.concurrent_games)

        print(f"TrainingOrchestrator setup complete:")
        print(f"  Concurrent games : {self.cfg.concurrent_games}")
        print(f"  Batch size       : {self.cfg.max_batch_size}")
        print(f"  Training step    : {self.training_step}")
        print(f"  GPU available    : {torch.cuda.is_available()}")

    def _build_logger(self) -> SummaryWriter:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f"logs/{self.cfg.run_name}{ts}"
        return SummaryWriter(log_dir)

    # ------------------------------------------------------------------
    # 1️⃣  COLLECT phase
    # ------------------------------------------------------------------

    async def collect(self) -> None:
        """
        Start the continuous game-collection loop.
        This runs forever (or until :meth:`stop_training` is called) and
        calls :meth:`train_step` whenever the global buffer has data.
        """
        self.training_active = True
        print("COLLECT phase started – running continuous games...")

        async def _on_game_done(result: GameResult):
            self.games_completed += 1
            if self.global_buffer and self.global_buffer.available_gameplay_batch():
                await self._train_step()

        await self.env_manager.run_continuously(self.agent_manager, _on_game_done)

    # ------------------------------------------------------------------
    # 2️⃣  TRAIN phase
    # ------------------------------------------------------------------

    async def _train_step(self) -> None:
        """Perform a single training step (called automatically during collect)."""
        if not self.global_buffer or not self.global_buffer.available_gameplay_batch():
            return

        batch = self.global_buffer.read_gameplay_batch()
        combat_batch = []
        if hasattr(self.global_buffer, "available_combat_batch") and self.global_buffer.available_combat_batch():
            combat_batch = self.global_buffer.read_combat_batch()

        self.trainer.train_network(
            batch=batch,
            combats=combat_batch,
            agent=self.base_agent.model,
            train_step=self.training_step,
            summary_writer=self.summary_writer,
        )
        self.training_step += 1

        # Periodic evaluation and checkpointing
        if self.training_step % self.cfg.evaluation_interval == 0:
            await self.evaluate()

        if self.training_step % self.cfg.save_interval == 0:
            self.save_checkpoint()

    def train_step(self) -> bool:
        """
        Explicit TRAIN step.  Useful when driving the lifecycle manually.
        Returns True if training actually occurred.
        """
        if not self.global_buffer or not self.global_buffer.available_gameplay_batch():
            return False
        asyncio.create_task(self._train_step())
        return True

    # ------------------------------------------------------------------
    # 3️⃣  SYNC phase
    # ------------------------------------------------------------------

    def sync_weights(self) -> None:
        """
        Distribute the latest trained weights to the active collection
        agents so they immediately benefit from the new policy.
        """
        if not self.base_agent:
            return
        new_weights = self.base_agent.get_weights()
        self.current_weights = copy.deepcopy(new_weights)
        for agent in self._training_agents:
            agent.update_weights(new_weights)
        print(f"SYNC: distributed weights to {len(self._training_agents)} agent(s)")

    # ------------------------------------------------------------------
    # 4️⃣  EVALUATE phase
    # ------------------------------------------------------------------

    async def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation games between the current (new) model and the
        previously saved (old) model.  Keep the better-performing weights.

        Returns a dict with ``new_placement`` and ``old_placement``.
        """
        print(f"\nEVALUATE at step {self.training_step}")

        eval_base = MuZeroAgent(
            action_size=3,
            action_limits=[7, 37, 10],
            obs_size=config.OBSERVATION_SIZE,
            simulations=config.NUM_SIMULATIONS,
            global_buffer=self.global_buffer,
            weights=copy.deepcopy(self.base_agent.get_weights()),
        )
        eval_old = MuZeroAgent(
            action_size=3,
            action_limits=[7, 37, 10],
            obs_size=config.OBSERVATION_SIZE,
            simulations=config.NUM_SIMULATIONS,
            global_buffer=self.global_buffer,
            weights=copy.deepcopy(self.current_weights),
        )
        random_agent = RandomAgent("EvalRandom")
        cultist_agent = CultistAgent()
        divine_agent = DivineAgent()

        eval_configs = [
            (eval_base, 1),
            (eval_old, 1),
            (random_agent, 4),
            (cultist_agent, 1),
            (divine_agent, 1),
        ]
        eval_mgr, _ = create_custom_agent_setup(
            eval_configs,
            max_batch_size=self.cfg.max_batch_size,
            batch_timeout_ms=self.cfg.batch_timeout_ms,
            gpu_memory_fraction=self.cfg.gpu_memory_fraction,
        )

        eval_env_mgr = _ParallelEnvManager(self.cfg.evaluation_concurrent)
        results = await eval_env_mgr.run_fixed_games(eval_mgr, self.cfg.evaluation_games)

        base_placements, old_placements = [], []
        for r in results:
            mapping = r.agent_mapping
            for pid, placement in r.placements.items():
                at = mapping.get(pid)
                if at == type(eval_base):
                    base_placements.append(placement)
                elif at == type(eval_old):
                    old_placements.append(placement)

        base_mean = float(np.mean(base_placements)) if base_placements else 8.0
        old_mean = float(np.mean(old_placements)) if old_placements else 8.0

        if self.summary_writer:
            self.summary_writer.add_scalar("evaluation/new_model", base_mean, self.training_step)
            self.summary_writer.add_scalar("evaluation/old_model", old_mean, self.training_step)

        print(f"  New model placement: {base_mean:.2f}  |  Old model: {old_mean:.2f}")

        if base_mean < old_mean:
            print("  ✓ Model improved – updating weights & clearing buffers.")
            self.current_weights = copy.deepcopy(self.base_agent.get_weights())
            self.save_checkpoint()
            if self.global_buffer:
                if hasattr(self.global_buffer, "clear_gameplay_buffer"):
                    self.global_buffer.clear_gameplay_buffer()
                if hasattr(self.global_buffer, "clear_combat_buffer"):
                    self.global_buffer.clear_combat_buffer()
            self.sync_weights()

        return {"new_placement": base_mean, "old_placement": old_mean}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_checkpoint(self) -> None:
        os.makedirs("./checkpoint", exist_ok=True)
        path = f"./checkpoint/checkpoint_{self.training_step}"
        if self.base_agent is not None:
            torch.save(self.base_agent.model.state_dict(), path)
            print(f"Checkpoint saved at step {self.training_step}")

    def load_checkpoint(self, step: int) -> bool:
        path = f"./checkpoint/checkpoint_{step}"
        if not os.path.isfile(path):
            return False
        state = torch.load(path)
        if self.base_agent is not None:
            self.base_agent.model.load_state_dict(state)
        self.training_step = step
        return True

    # ------------------------------------------------------------------
    # High-level modes
    # ------------------------------------------------------------------

    async def run(self, max_steps: int = 1_000_000) -> None:
        """
        Full training loop orchestrating all lifecycle phases.

        COLLECT runs in the background; TRAIN is triggered whenever the
        buffer has data; SYNC follows weight improvement; EVALUATE runs
        periodically.
        """
        if self.env_manager is None:
            self.setup()

        self.training_active = True

        async def _internal_callback(result: GameResult):
            self.games_completed += 1
            if self.global_buffer and self.global_buffer.available_gameplay_batch():
                await self._train_step()

        collect_task = asyncio.create_task(
            self.env_manager.run_continuously(self.agent_manager, _internal_callback)
        )

        try:
            while self.training_active and self.training_step < max_steps:
                await asyncio.sleep(1.0)
                if self.training_step % 100 == 0 and self.training_step > 0:
                    print(f"  step={self.training_step}  games={self.games_completed}")
        except asyncio.CancelledError:
            pass
        finally:
            self.training_active = False
            self.env_manager.stop()
            await collect_task
            if self.summary_writer:
                self.summary_writer.close()

    def stop_training(self):
        """Gracefully stop the training loop."""
        self.training_active = False
        if self.env_manager:
            self.env_manager.stop()

    async def run_single_episode(self) -> GameResult:
        """
        Run a single episode for debugging.

        Replaces the old ``train_single.py`` workflow.
        """
        if self.agent_manager is None:
            self.setup()
        worker = _GameWorker(0)
        return await worker.run_game(self.agent_manager, return_placements=True)

    async def run_parallel_demo(self, num_episodes: int = 5) -> List[GameResult]:
        """
        Run *num_episodes* games in parallel.

        Replaces the old ``train_parallel.py`` demo workflow.
        """
        if self.agent_manager is None:
            self.setup()
        mgr = _ParallelEnvManager(min(self.cfg.concurrent_games, num_episodes))
        return await mgr.run_fixed_games(self.agent_manager, num_episodes)

    async def run_evaluation(self, num_games: int) -> List[GameResult]:
        """Run a standalone evaluation session."""
        if self.agent_manager is None:
            self.setup()
        mgr = _ParallelEnvManager(self.cfg.evaluation_concurrent)
        return await mgr.run_fixed_games(self.agent_manager, num_games)

    def cleanup(self):
        """Release resources (writer, etc.)."""
        if self.summary_writer:
            self.summary_writer.close()
        self.training_active = False


# ---------------------------------------------------------------------------
# Convenience factory / helpers
# ---------------------------------------------------------------------------

def create_orchestrator(config: Optional[TrainingConfig] = None) -> TrainingOrchestrator:
    return TrainingOrchestrator(config)


async def quick_evaluation(num_games: int = 8, concurrent: int = 2) -> List[GameResult]:
    """Quick evaluation with default agents."""
    cfg = TrainingConfig(
        concurrent_games=concurrent,
        evaluation_games=num_games,
        evaluation_concurrent=concurrent,
        max_batch_size=8,
    )
    orch = TrainingOrchestrator(cfg)
    orch.setup()
    mgr = _ParallelEnvManager(concurrent)
    results = await mgr.run_fixed_games(orch.agent_manager, num_games)

    agent_stats: Dict[str, List[int]] = defaultdict(list)
    for r in results:
        for pid, placement in r.placements.items():
            at = r.agent_mapping.get(pid)
            name = at.__name__ if at and hasattr(at, "__name__") else str(at)
            agent_stats[name].append(placement)

    print(f"\n=== Evaluation Summary ({len(results)} games) ===")
    for name, placements in agent_stats.items():
        print(f"  {name}: {np.mean(placements):.2f} avg placement ({len(placements)} games)")
    return results
