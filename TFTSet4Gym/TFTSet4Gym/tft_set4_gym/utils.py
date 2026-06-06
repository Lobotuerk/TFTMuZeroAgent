from . import config as config
import numpy as np
from functools import wraps
from time import time
from .stats import COST

# Observation layout (new flat format, total = 1804):
#   board_champions:  [0      : 1624)  (58, 4, 7)
#   board_stars:      [1624  : 1652)  (1, 4, 7)
#   board_chosen:     [1652  : 1680)  (1, 4, 7)
#   bench_champions:  [1680  : 1738)  (58,)
#   health:           [1738  : 1739)  (1,)
#   turns_for_combat: [1739  : 1740)  (1,)
#   level:            [1740  : 1741)  (1,)
#   round:            [1741  : 1742)  (1,)
#   exp_to_level:     [1742  : 1743)  (1,)
#   gold:             [1743  : 1744)  (1,)
#   streak:           [1744  : 1745)  (1,)
#   shop_champions:   [1745  : 1803)  (58,)
#   shop_chosen:      [1803  : 1804)  (1,)

BOARD_CHAMPIONS_SLICE = slice(0, 1624)
BOARD_STARS_SLICE = slice(1624, 1652)
BOARD_CHOSEN_SLICE = slice(1652, 1680)
BENCH_SLICE = slice(1680, 1738)
HEALTH_IDX = 1738
TFC_IDX = 1739
LEVEL_IDX = 1740
ROUND_IDX = 1741
EXP_IDX = 1742
GOLD_IDX = 1743
STREAK_IDX = 1744
SHOP_SLICE = slice(1745, 1803)
SHOP_CHOSEN_IDX = 1803

OBSERVATION_SIZE = 1804


def get_field_indices_safe(field_name, default_start=0, default_end=1):
    """Get field indices with fallback for backward compatibility."""
    try:
        from .observation_schema import get_field_indices
        return get_field_indices(field_name)
    except (ImportError, KeyError):
        # Fallback to hardcoded values if schema not available
        hardcoded_map = {
            'board_champions': (0, 1624),
            'board_stars': (1624, 1652),
            'board_chosen': (1652, 1680),
            'bench_champions': (1680, 1738),
            'health': (1738, 1739),
            'turns_for_combat': (1739, 1740),
            'level': (1740, 1741),
            'round': (1741, 1742),
            'exp_to_level': (1742, 1743),
            'gold': (1743, 1744),
            'streak': (1744, 1745),
            'shop_champions': (1745, 1803),
            'shop_chosen': (1803, 1804),
        }
        return hardcoded_map.get(field_name, (default_start, default_end))



def champ_binary_encode(n):
    return list(np.unpackbits(np.array([n], np.uint8))[2:8])

def champ_binary_decode(array):
    temp = list(array.copy().astype(int))
    temp.insert(0, 0)
    temp.insert(0, 0)
    return np.packbits(temp, axis=-1)[0]

def item_binary_encode(n):
    return list(np.unpackbits(np.array([n], np.uint8))[2:8])

def champ_one_hot_encode(n):
    return np.eye(config.MAX_CHAMPION_IN_SET)[n]

def item_one_hot_encode(n):
    return np.eye(9)[n]

def one_hot_encode_number(number, depth):
    return np.eye(depth)[number]


def timed(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        start = time()
        result = f(*args, **kwds)
        elapsed = time() - start
        print(f'{f.__name__} took {elapsed} seconds to finish')
        return result

    return wrapper


def decode_action(str_actions):
    actions = []
    for str_action in str_actions:
        num_items = str_action.count("_")
        split_action = str_action.split("_")
        element_list = [0, 0, 0, 0]
        for i in range(num_items + 1):
            element_list[i] = int(split_action[i])
        actions.append(np.asarray(element_list))
    return np.asarray(actions)


def x_y_to_1d_coord(x1, y1):
    if y1 == -1:
        return x1 + 28
    else:
        return 7 * y1 + x1
    
def player_map_from_obs(observation):
    player_map = {}
    player_map["gold"] = gold_from_obs(observation)
    player_map["shop"], player_map["chosen_shop"] = units_in_shop_from_obs(observation)
    player_map["board"] = board_from_obs(observation)
    player_map["bench"] = bench_from_obs(observation)
    player_map["level"] = level_from_obs(observation)
    player_map['hp'] = hp_from_obs(observation)
    player_map['round'] = round_from_obs(observation)
    player_map['turns_for_combat'] = t_f_c_from_obs(observation)
    player_map['exp_to_level'] = exp_to_level_from_obs(observation)
    player_map['streak'] = streak_from_obs(observation)
    return player_map

def streak_from_obs(observation):
    if observation.ndim > 1:
        observation = observation.flatten()
    start, end = get_field_indices_safe('streak', STREAK_IDX, STREAK_IDX + 1)
    return observation[start]

def gold_from_obs(observation):
    if observation.ndim > 1:
        observation = observation.flatten()
    start, end = get_field_indices_safe('gold', GOLD_IDX, GOLD_IDX + 1)
    return observation[start]

def exp_to_level_from_obs(observation):
    if observation.ndim > 1:
        observation = observation.flatten()
    start, end = get_field_indices_safe('exp_to_level', EXP_IDX, EXP_IDX + 1)
    return observation[start]

def hp_from_obs(observation):
    if observation.ndim > 1:
        observation = observation.flatten()
    start, end = get_field_indices_safe('health', HEALTH_IDX, HEALTH_IDX + 1)
    return observation[start]

def round_from_obs(observation):
    if observation.ndim > 1:
        observation = observation.flatten()
    start, end = get_field_indices_safe('round', ROUND_IDX, ROUND_IDX + 1)
    return observation[start]

def t_f_c_from_obs(observation):
    if observation.ndim > 1:
        observation = observation.flatten()
    start, end = get_field_indices_safe('turns_for_combat', TFC_IDX, TFC_IDX + 1)
    return observation[start]

def units_in_shop_from_obs(observation):
    if observation.ndim > 1:
        observation = observation.flatten()
    shop_start, shop_end = get_field_indices_safe('shop_champions', SHOP_SLICE.start, SHOP_SLICE.stop)
    chosen_idx, _ = get_field_indices_safe('shop_chosen', SHOP_CHOSEN_IDX, SHOP_CHOSEN_IDX + 1)

    units = observation[shop_start:shop_end]
    chosen_val = observation[chosen_idx]
    if int(chosen_val) > 0:
        chosen = list(COST.keys())[int(chosen_val) + 1] + "_chosen"
    else:
        chosen = ""
    parsed_units = []
    for i, count in enumerate(units):
        if count > 0:
            for _ in range(int(count)):
                parsed_units.append(list(COST.keys())[i + 1])
    return parsed_units, chosen

def _board_3d(observation):
    """Extract board fields and reshape to (58,4,7), (4,7), (4,7)."""
    if observation.ndim > 1:
        observation = observation.flatten()
    board_flat = observation[BOARD_CHAMPIONS_SLICE]
    stars_flat = observation[BOARD_STARS_SLICE]
    chosen_flat = observation[BOARD_CHOSEN_SLICE]
    return (
        board_flat.reshape(58, 4, 7),
        stars_flat.reshape(4, 7),
        chosen_flat.reshape(4, 7),
    )

def board_from_obs(observation):
    board, stars, chosen = _board_3d(observation)
    champs = []
    for i, unit_board in enumerate(board):
        indexes = np.where(unit_board == 1.0)
        if len(indexes[0]) > 0:
            champ = {
                "name": list(COST.keys())[i + 1],
                "id": i,
                "pos_y": int(indexes[0][0]),
                "pos_x": int(indexes[1][0]),
                "stars": stars[indexes[0], indexes[1]][0],
                "chosen": chosen[indexes[0], indexes[1]][0] > 0.0,
            }
            champs.append(champ)
    return champs

def bench_from_obs(observation):
    if observation.ndim > 1:
        observation = observation.flatten()
    bench = observation[BENCH_SLICE]
    bench_list = []
    for i, n in enumerate(bench):
        if n > 0:
            for _ in range(int(n)):
                bench_list.append(list(COST.keys())[i + 1])
    return bench_list

def champ_id_from_name(champ_name):
    return (list(COST.keys()).index(champ_name)) - 1

def level_from_obs(observation):
    if observation.ndim > 1:
        observation = observation.flatten()
    start, end = get_field_indices_safe('level', LEVEL_IDX, LEVEL_IDX + 1)
    return observation[start]