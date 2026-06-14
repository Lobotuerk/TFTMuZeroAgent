import sys
import numpy as np
# AI RELATED VALUES START HERE

# GIL / Free-Threading detection (Python 3.13+)
IS_GIL_DISABLED: bool = not getattr(sys, '_is_gil_enabled', lambda: True)()
"""True when running on a free-threaded (no-GIL) Python build."""

FORCE_THREADING_ENV_MANAGER: bool = True
"""Set to True to prefer thread-based env managers even when GIL is active."""

#### MODEL SET UP ####
HIDDEN_STATE_SIZE = 2048
NUM_RNN_CELLS = 8
LSTM_SIZE = int(HIDDEN_STATE_SIZE / (NUM_RNN_CELLS * 2))
RNN_SIZES = [LSTM_SIZE] * NUM_RNN_CELLS
LAYER_HIDDEN_SIZE = 512
ROOT_DIRICHLET_ALPHA = 1.0
ROOT_EXPLORATION_FRACTION = 0.2
MINIMUM_REWARD = -300.0
MAXIMUM_REWARD = 300.0
PB_C_BASE = 19652
PB_C_INIT = 1.25
DISCOUNT = 0.97
TRAINING_STEPS = 1e10

# Resolve observation size dynamically based on installed/imported TFTSet4Gym schema
try:
    from TFTSet4Gym.tft_set4_gym.observation_schema import get_observation_schema
    OBSERVATION_SIZE = get_observation_schema("current_player").total_size
except Exception:
    OBSERVATION_SIZE = 28946  # Fallback

OBSERVATION_TIME_STEPS = 1
OBSERVATION_TIME_STEP_INTERVAL = 1
ACTION_ENCODING_SIZE = 55  # sum(ACTION_DIM) = 8+37+10; matches 3-block variable-dim encoding
ACTION_CONCAT_SIZE = 55
ACTION_DIM = [8, 37, 10]

# ACTION_DIM = 10
ENCODER_NUM_STEPS = 601
MAX_GRAD_NORM = 5.0

### TIME RELATED VALUES ###
ACTIONS_PER_TURN = 25
CONCURRENT_GAMES = 4
COLLECT_GAMES_PER_BATCH = 4
NUM_PLAYERS = 8
NUM_SIMULATIONS = 50
UNROLL_STEPS = 5
EVALUATION_GAMES = 8
EVALUATION_CONCURRENT_GAMES = 4

### TRAINING ###
BATCH_SIZE = 128
INIT_LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = int(350e3)
LR_DECAY_FUNCTION = 0.1
SYNC_STEPS = 200

DEBUG = False
CHECKPOINT_STEPS = 200
REPLAY_BUFFER_SIZE = 10000
GAMEPLAY_BUFFER_PATH = './data/gameplay/'
COMBAT_BUFFER_PATH = './data/combats/'
RESULTS_PATH = './Checkpoints'

LOG_COMBAT = False
LOGMESSAGES = True

BATCHED_INFERENCE_THRESHOLD = 64
