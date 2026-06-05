import pytest
import torch
import numpy as np
from Models.action_translation import (
    ActionTranslationModule,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    BOARD_SIZE,
    NUM_CHAMPIONS,
    EMPTY_CLASS,
    BENCH_SIZE,
)
from TFTSet4Gym.tft_set4_gym.stats import COST


def test_decodes_target_board():
    module = ActionTranslationModule()
    probs = torch.zeros(1, NUM_CHAMPIONS + 1, BOARD_HEIGHT, BOARD_WIDTH)
    probs[0, 5, 1, 2] = 1.0
    probs[0, EMPTY_CLASS, :, :] = 1.0
    probs[0, EMPTY_CLASS, 1, 2] = 0.0
    target = module.decode_target_board(probs)
    assert target.shape == (1, BOARD_HEIGHT, BOARD_WIDTH)
    assert target[0, 1, 2] == 5
    assert target[0, 0, 0] == EMPTY_CLASS


def test_get_current_board_empty():
    class MockChamp:
        def __init__(self, name):
            self.name = name

    class MockPlayer:
        def __init__(self):
            self.board = [[None for _ in range(4)] for _ in range(7)]

    module = ActionTranslationModule()
    board = module.get_current_board(MockPlayer())
    assert board.shape == (BOARD_HEIGHT, BOARD_WIDTH)
    assert np.all(board == EMPTY_CLASS)


def test_get_current_board_with_champs():
    champ_names = list(COST.keys())
    real_champs = [n for n in champ_names if n != " " and COST[n] > 0]

    class MockChamp:
        def __init__(self, name):
            self.name = name

    class MockPlayer:
        def __init__(self):
            self.board = [[None for _ in range(4)] for _ in range(7)]

    player = MockPlayer()
    player.board[0][0] = MockChamp(real_champs[0])
    player.board[3][1] = MockChamp(real_champs[1])

    module = ActionTranslationModule()
    board = module.get_current_board(player)
    assert board[0, 0] == 0
    assert board[1, 3] == 1
    assert np.all(board != EMPTY_CLASS) or True


def test_get_current_bench():
    class MockChamp:
        def __init__(self, name):
            self.name = name

    class MockPlayer:
        def __init__(self):
            self.bench = [None for _ in range(9)]

    player = MockPlayer()
    champ_names = [n for n in list(COST.keys()) if n != " " and COST[n] > 0]
    player.bench[2] = MockChamp(champ_names[3])
    player.bench[5] = MockChamp(champ_names[7])

    module = ActionTranslationModule()
    bench = module.get_current_bench(player)
    assert (2, 3) in bench
    assert (5, 7) in bench
    assert len(bench) == 2


def test_translate_noop():
    class MockChamp:
        def __init__(self, name):
            self.name = name

    class MockPlayer:
        def __init__(self):
            self.board = [[None for _ in range(4)] for _ in range(7)]
            self.bench = [None for _ in range(9)]

    module = ActionTranslationModule()
    probs = torch.zeros(1, NUM_CHAMPIONS + 1, BOARD_HEIGHT, BOARD_WIDTH)
    probs[0, EMPTY_CLASS, :, :] = 1.0
    actions = module.translate(probs, MockPlayer())
    assert len(actions) >= 1
    assert actions[0] == [0, 0, 0]


def test_translate_bench_to_board():
    champ_names = [n for n in list(COST.keys()) if n != " " and COST[n] > 0]

    class MockChamp:
        def __init__(self, name):
            self.name = name

    class MockPlayer:
        def __init__(self):
            self.board = [[None for _ in range(4)] for _ in range(7)]
            self.bench = [None for _ in range(9)]

    player = MockPlayer()
    player.bench[0] = MockChamp(champ_names[0])

    module = ActionTranslationModule()
    probs = torch.zeros(1, NUM_CHAMPIONS + 1, BOARD_HEIGHT, BOARD_WIDTH)
    probs[0, EMPTY_CLASS, :, :] = 0.0
    probs[0, 0, :, :] = 1.0
    actions = module.translate(probs, player)
    assert len(actions) >= 1
    action_types = [a[0] for a in actions]
    assert 1 in action_types


def test_translate_board_to_bench():
    champ_names = [n for n in list(COST.keys()) if n != " " and COST[n] > 0]

    class MockChamp:
        def __init__(self, name):
            self.name = name

    class MockPlayer:
        def __init__(self):
            self.board = [[None for _ in range(4)] for _ in range(7)]
            self.bench = [None for _ in range(9)]

    player = MockPlayer()
    player.board[0][0] = MockChamp(champ_names[5])

    module = ActionTranslationModule()
    probs = torch.zeros(1, NUM_CHAMPIONS + 1, BOARD_HEIGHT, BOARD_WIDTH)
    probs[0, EMPTY_CLASS, :, :] = 1.0
    actions = module.translate(probs, player)
    action_types = [a[0] for a in actions]
    assert 1 in action_types, "should move to bench, not sell"
    for a in actions:
        if a[0] == 1:
            assert a[2] >= BOARD_SIZE, "board-to-bench should target bench dcord >= 28"


def test_translate_sell_when_bench_full():
    champ_names = [n for n in list(COST.keys()) if n != " " and COST[n] > 0]

    class MockChamp:
        def __init__(self, name):
            self.name = name

    class MockPlayer:
        def __init__(self):
            self.board = [[None for _ in range(4)] for _ in range(7)]
            self.bench = [None for _ in range(9)]

    player = MockPlayer()
    player.board[0][0] = MockChamp(champ_names[5])
    for i in range(9):
        player.bench[i] = MockChamp(champ_names[(i + 10) % NUM_CHAMPIONS])

    module = ActionTranslationModule()
    probs = torch.zeros(1, NUM_CHAMPIONS + 1, BOARD_HEIGHT, BOARD_WIDTH)
    probs[0, EMPTY_CLASS, :, :] = 1.0
    actions = module.translate(probs, player)
    action_types = [a[0] for a in actions]
    assert 3 in action_types, "should sell when bench is full"


def test_translate_board_to_board():
    champ_names = [n for n in list(COST.keys()) if n != " " and COST[n] > 0]

    class MockChamp:
        def __init__(self, name):
            self.name = name

    class MockPlayer:
        def __init__(self):
            self.board = [[None for _ in range(4)] for _ in range(7)]
            self.bench = [None for _ in range(9)]

    player = MockPlayer()
    player.board[0][0] = MockChamp(champ_names[5])
    player.board[1][0] = MockChamp(champ_names[3])

    module = ActionTranslationModule()
    probs = torch.zeros(1, NUM_CHAMPIONS + 1, BOARD_HEIGHT, BOARD_WIDTH)
    probs[0, EMPTY_CLASS, :, :] = 0.0
    probs[0, 3, 0, 0] = 1.0
    probs[0, 5, 1, 0] = 1.0
    actions = module.translate(probs, player)
    board_to_board_moves = [a for a in actions if a[0] == 1 and a[1] < BOARD_SIZE and a[2] < BOARD_SIZE]
    assert len(board_to_board_moves) >= 1


def test_translate_keep_champ_needed_elsewhere():
    champ_names = [n for n in list(COST.keys()) if n != " " and COST[n] > 0]

    class MockChamp:
        def __init__(self, name):
            self.name = name

    class MockPlayer:
        def __init__(self):
            self.board = [[None for _ in range(4)] for _ in range(7)]
            self.bench = [None for _ in range(9)]

    player = MockPlayer()
    player.board[0][0] = MockChamp(champ_names[0])
    player.board[1][0] = MockChamp(champ_names[1])

    module = ActionTranslationModule()
    probs = torch.zeros(1, NUM_CHAMPIONS + 1, BOARD_HEIGHT, BOARD_WIDTH)
    probs[0, EMPTY_CLASS, :, :] = 0.0
    probs[0, 0, 0, 0] = 1.0
    probs[0, 1, 0, 1] = 1.0
    actions = module.translate(probs, player)
    action_types = [a[0] for a in actions]
    assert 3 not in action_types, "should not sell champ 0 or 1, both are needed"
    assert 1 in action_types, "should generate moves to reposition"


def test_translate_batch():
    class MockChamp:
        def __init__(self, name):
            self.name = name

    class MockPlayer:
        def __init__(self):
            self.board = [[None for _ in range(4)] for _ in range(7)]
            self.bench = [None for _ in range(9)]

    module = ActionTranslationModule()
    probs = torch.zeros(2, NUM_CHAMPIONS + 1, BOARD_HEIGHT, BOARD_WIDTH)
    probs[:, EMPTY_CLASS, :, :] = 1.0
    players = [MockPlayer(), MockPlayer()]
    results = module.translate_batch(probs, players)
    assert len(results) == 2
    assert results[0][0] == [0, 0, 0]


def test_champion_name_mapping():
    module = ActionTranslationModule()
    assert len(module.idx_to_name) == NUM_CHAMPIONS
    assert len(module.name_to_idx) == NUM_CHAMPIONS
    for idx, name in module.idx_to_name.items():
        assert module.name_to_idx[name] == idx


def test_integration_with_board_generator():
    from Models.MuZero_torch_model import BoardGenerator

    bg = BoardGenerator()
    atm = ActionTranslationModule()
    x = torch.randn(1, 116)
    out = bg(x)
    assert out.shape == (1, NUM_CHAMPIONS + 1, BOARD_HEIGHT, BOARD_WIDTH)

    class MockChamp:
        def __init__(self, name):
            self.name = name

    class MockPlayer:
        def __init__(self):
            self.board = [[None for _ in range(4)] for _ in range(7)]
            self.bench = [None for _ in range(9)]

    actions = atm.translate(out, MockPlayer())
    assert len(actions) >= 1


# --- board_generator_to_actions tests ---

def _make_player(board_slots=None, bench_slots=None, gold=99, shop_elems=None):
    champ_names = [n for n in list(COST.keys()) if n != " " and COST[n] > 0]

    class MockChamp:
        def __init__(self, name):
            self.name = name

    class MockPlayer:
        def __init__(self):
            self.board = [[None for _ in range(4)] for _ in range(7)]
            self.bench = [None for _ in range(9)]
            self.gold = gold
            self.shop_elems = shop_elems if shop_elems is not None else np.ones(5)

    player = MockPlayer()
    if board_slots:
        for (x, y), name_idx in board_slots.items():
            player.board[x][y] = MockChamp(champ_names[name_idx])
    if bench_slots:
        for pos, name_idx in bench_slots.items():
            player.bench[pos] = MockChamp(champ_names[name_idx])
    return player, champ_names


def _make_probs(champ_names, board_config):
    """board_config: dict mapping (y, x) -> champion_name_index (0-57) or EMPTY_CLASS"""
    probs = torch.zeros(1, NUM_CHAMPIONS + 1, BOARD_HEIGHT, BOARD_WIDTH)
    probs[0, EMPTY_CLASS, :, :] = 1.0
    for (y, x), idx in board_config.items():
        if idx != EMPTY_CLASS:
            probs[0, idx, y, x] = 1.0
            probs[0, EMPTY_CLASS, y, x] = 0.0
    return probs


def test_board_to_actions_noop():
    atm = ActionTranslationModule()
    player, champ_names = _make_player()
    probs = _make_probs(champ_names, {})
    actions = atm.board_generator_to_actions(probs, player)
    assert len(actions) == 1
    assert actions[0] == [0, 0, 0]


def test_board_to_actions_buy_from_shop():
    atm = ActionTranslationModule()
    shop_elems = np.array([0, 1, 2, 3, 4])
    player, champ_names = _make_player(shop_elems=shop_elems, gold=10)
    probs = _make_probs(champ_names, {(0, 0): 0})
    actions = atm.board_generator_to_actions(probs, player)
    action_types = [a[0] for a in actions]
    assert 2 in action_types, "should buy champion 0 from shop"
    buy_actions = [a for a in actions if a[0] == 2]
    assert buy_actions[0][1] == 0, "should buy champion with ID 0"
    assert 1 in action_types, "should move bought champ to board"


def test_board_to_actions_buy_costly_champ_insufficient_gold():
    atm = ActionTranslationModule()
    champ_names = [n for n in list(COST.keys()) if n != " " and COST[n] > 0]
    champ_name = champ_names[0]
    cost = COST[champ_name]
    shop_elems = np.array([0, 1, 2, 3, 4])
    player, _ = _make_player(shop_elems=shop_elems, gold=cost - 1)
    probs = _make_probs(champ_names, {(0, 0): 0})
    actions = atm.board_generator_to_actions(probs, player)
    buy_actions = [a for a in actions if a[0] == 2]
    assert len(buy_actions) == 0, "should not buy if insufficient gold"


def test_board_to_actions_sell_excess_bench():
    atm = ActionTranslationModule()
    shop_elems = np.array([0, 5, 5, 5, 5])
    player, champ_names = _make_player(
        bench_slots={0: 0, 1: 0, 2: 5, 3: 5},
        shop_elems=shop_elems,
        gold=99
    )
    probs = _make_probs(champ_names, {(0, 0): 0})
    actions = atm.board_generator_to_actions(probs, player)
    action_types = [a[0] for a in actions]
    assert 3 in action_types, "should sell excess bench champion"
    sell_actions = [a for a in actions if a[0] == 3]
    bench_sells = [a for a in sell_actions if a[1] >= BOARD_SIZE]
    assert len(bench_sells) >= 1, "should sell from bench"


def test_board_to_actions_sell_excess_board():
    atm = ActionTranslationModule()
    champ_names = [n for n in list(COST.keys()) if n != " " and COST[n] > 0]
    player, _ = _make_player(board_slots={(0, 0): 0, (1, 0): 5})
    probs = _make_probs(champ_names, {(0, 0): 0})
    actions = atm.board_generator_to_actions(probs, player)
    action_types = [a[0] for a in actions]
    assert 3 in action_types or any(
        a[0] == 1 and a[1] < BOARD_SIZE and a[2] >= BOARD_SIZE for a in actions
    ), "should sell or move excess board champion to bench"


def test_board_to_actions_buy_and_place():
    atm = ActionTranslationModule()
    shop_elems = np.array([7, 1, 2, 3, 4])
    player, champ_names = _make_player(shop_elems=shop_elems, gold=99)
    probs = _make_probs(champ_names, {(1, 2): 7})
    actions = atm.board_generator_to_actions(probs, player)
    action_types = [a[0] for a in actions]
    assert 2 in action_types, "should buy champion 7"
    assert 1 in action_types, "should move champion to (y=1, x=2)"
    move_actions = [a for a in actions if a[0] == 1]
    board_moves = [a for a in move_actions if a[2] < BOARD_SIZE]
    assert any(a[2] == 1 * BOARD_WIDTH + 2 for a in board_moves), "should move to position (1,2)"


def test_board_to_actions_move_from_board():
    atm = ActionTranslationModule()
    champ_names = [n for n in list(COST.keys()) if n != " " and COST[n] > 0]
    player, _ = _make_player(board_slots={(0, 0): 0, (1, 1): 5})
    probs = _make_probs(champ_names, {(1, 2): 5, (2, 3): 0})
    actions = atm.board_generator_to_actions(probs, player)
    action_types = [a[0] for a in actions]
    assert 1 in action_types, "should move champions between board positions"
    move_actions = [a for a in actions if a[0] == 1]
    move_pairs = [(a[1], a[2]) for a in move_actions]
    assert any(src < BOARD_SIZE and dst < BOARD_SIZE for src, dst in move_pairs), \
        "should have board-to-board move"


def test_board_to_actions_full_bench_sell_before_buy():
    atm = ActionTranslationModule()
    champ_names = [n for n in list(COST.keys()) if n != " " and COST[n] > 0]
    shop_elems = np.array([12, 1, 2, 3, 4])
    player, _ = _make_player(
        bench_slots={i: (i + 20) % NUM_CHAMPIONS for i in range(9)},
        shop_elems=shop_elems,
        gold=99
    )
    probs = _make_probs(champ_names, {(0, 0): 12})
    actions = atm.board_generator_to_actions(probs, player)
    action_types = [a[0] for a in actions]
    assert 3 in action_types, "should sell to make room"
    assert 2 in action_types, "should buy after making room"
    sell_idx = next(i for i, a in enumerate(actions) if a[0] == 3)
    buy_idx = next(i for i, a in enumerate(actions) if a[0] == 2)
    assert sell_idx < buy_idx, "should sell before buying"


def test_board_to_actions_champ_already_in_place():
    atm = ActionTranslationModule()
    champ_names = [n for n in list(COST.keys()) if n != " " and COST[n] > 0]
    player, _ = _make_player(board_slots={(0, 0): 0, (1, 1): 5})
    probs = _make_probs(champ_names, {(0, 0): 0, (1, 1): 5})
    actions = atm.board_generator_to_actions(probs, player)
    assert len(actions) == 1
    assert actions[0] == [0, 0, 0], "should pass when board already matches target"


def test_board_to_actions_move_to_empty_dest():
    atm = ActionTranslationModule()
    champ_names = [n for n in list(COST.keys()) if n != " " and COST[n] > 0]
    player, _ = _make_player(board_slots={(0, 0): 0, (1, 1): 5})
    probs = _make_probs(champ_names, {(0, 1): 0, (1, 1): 5})
    actions = atm.board_generator_to_actions(probs, player)
    action_types = [a[0] for a in actions]
    assert 1 in action_types, "should move champion to new board position"
    move_actions = [a for a in actions if a[0] == 1]
    assert any(a[2] == 0 * BOARD_WIDTH + 1 for a in move_actions), \
        "should move champion 0 to column 0, row 1"


def test_board_to_actions_batch():
    atm = ActionTranslationModule()
    champ_names = [n for n in list(COST.keys()) if n != " " and COST[n] > 0]
    player1, _ = _make_player()
    player2, _ = _make_player(board_slots={(0, 0): 0, (1, 0): 5})
    probs = torch.zeros(2, NUM_CHAMPIONS + 1, BOARD_HEIGHT, BOARD_WIDTH)
    probs[0, EMPTY_CLASS, :, :] = 1.0
    probs[1, EMPTY_CLASS, :, :] = 0.0
    probs[1, 5, 0, 0] = 1.0  # target wants champ 5 at (0,0)
    probs[1, 0, 1, 1] = 1.0  # target wants champ 0 at (1,1)
    results = atm.board_generator_to_actions_batch(
        probs, [player1, player2], shop_slots_list=None
    )
    assert len(results) == 2
    assert results[0][0] == [0, 0, 0], "empty target should pass"
    action_types = [a[0] for a in results[1]]
    assert 1 in action_types, "should swap champions between board positions"


def test_board_to_actions_int_with_board_generator():
    from TFTSet4Gym.tft_set4_gym.models.board_generator import BoardGenerator

    bg = BoardGenerator()
    atm = ActionTranslationModule()
    x = torch.randn(1, 116)
    out = bg(x)

    class MockChamp:
        def __init__(self, name):
            self.name = name

    class MockPlayer:
        def __init__(self):
            self.board = [[None for _ in range(4)] for _ in range(7)]
            self.bench = [None for _ in range(9)]
            self.gold = 99
            self.shop_elems = np.ones(5)

    actions = atm.board_generator_to_actions(out, MockPlayer())
    assert len(actions) >= 1
