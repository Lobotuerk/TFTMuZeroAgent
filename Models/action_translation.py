import torch
import numpy as np
from TFTSet4Gym.tft_set4_gym.stats import COST

BOARD_HEIGHT = 4
BOARD_WIDTH = 7
BOARD_SIZE = BOARD_HEIGHT * BOARD_WIDTH
NUM_CHAMPIONS = 58
EMPTY_CLASS = NUM_CHAMPIONS
BENCH_SIZE = 9


def _parse_shop_slots(shop_slots, name_to_idx):
    """Parse shop slots into champion IDs, handling None, strings, and ints."""
    ids = []
    if shop_slots is None:
        return ids
    for slot in shop_slots:
        if slot is None or (isinstance(slot, str) and slot.strip() == ""):
            ids.append(None)
        elif isinstance(slot, str):
            base = slot.replace("_c", "")
            ids.append(name_to_idx.get(base))
        elif isinstance(slot, (int, np.integer)):
            ids.append(int(slot))
        else:
            ids.append(None)
    return ids


class ActionTranslationModule:
    def __init__(self):
        champ_names = list(COST.keys())
        self.idx_to_name = {}
        for i, name in enumerate(champ_names):
            if name != " " and COST[name] > 0:
                idx = i - 1
                self.idx_to_name[idx] = name
        self.name_to_idx = {v: k for k, v in self.idx_to_name.items()}

    def decode_target_board(self, board_probs: torch.Tensor) -> np.ndarray:
        target = board_probs.argmax(dim=1).cpu().numpy()
        return target

    def get_current_board(self, player) -> np.ndarray:
        current = np.full((BOARD_HEIGHT, BOARD_WIDTH), EMPTY_CLASS, dtype=int)
        for x in range(BOARD_WIDTH):
            for y in range(BOARD_HEIGHT):
                champ = player.board[x][y]
                if champ and champ.name in self.name_to_idx:
                    current[y, x] = self.name_to_idx[champ.name]
        return current

    def get_current_bench(self, player) -> list:
        bench = []
        for i in range(BENCH_SIZE):
            champ = player.bench[i]
            if champ and champ.name in self.name_to_idx:
                bench.append((i, self.name_to_idx[champ.name]))
        return bench

    def board_generator_to_actions(self, board_probs: torch.Tensor, player, shop_slots: list = None) -> list:
        target = self.decode_target_board(board_probs)
        if target.ndim == 2:
            target = target[None]
        batch_size = target.shape[0]
        results = []
        for i in range(batch_size):
            results.append(self._single_board_to_actions(target[i], player, shop_slots))
        return results if batch_size > 1 else results[0]

    def board_generator_to_actions_batch(self, board_probs: torch.Tensor, players: list, shop_slots_list: list = None) -> list:
        target = self.decode_target_board(board_probs)
        batch_size = target.shape[0]
        results = []
        for i in range(batch_size):
            slots = shop_slots_list[i] if shop_slots_list is not None and i < len(shop_slots_list) else None
            player = players[i] if i < len(players) else None
            if player is not None:
                results.append(self._single_board_to_actions(target[i], player, slots))
            else:
                results.append([[0, 0, 0]])
        return results

    def _single_board_to_actions(self, target_board: np.ndarray, player, shop_slots: list = None) -> list:
        current_board = self.get_current_board(player).copy()

        bench_state = [None] * BENCH_SIZE
        for pos, idx in self.get_current_bench(player):
            bench_state[pos] = idx

        shop_ids = _parse_shop_slots(shop_slots, self.name_to_idx)
        if not shop_ids:
            if hasattr(player, 'shop_elems'):
                shop_ids = [int(e) if int(e) < NUM_CHAMPIONS else None for e in player.shop_elems]
            else:
                shop_ids = []

        target_counts = {}
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                c_id = int(target_board[y, x])
                if c_id != EMPTY_CLASS:
                    target_counts[c_id] = target_counts.get(c_id, 0) + 1

        owned_counts = {}
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                c_id = int(current_board[y, x])
                if c_id != EMPTY_CLASS:
                    owned_counts[c_id] = owned_counts.get(c_id, 0) + 1
        for c_id in bench_state:
            if c_id is not None:
                owned_counts[c_id] = owned_counts.get(c_id, 0) + 1

        def _need(c_id):
            return target_counts.get(c_id, 0)
        def _have(c_id):
            return owned_counts.get(c_id, 0)

        actions = []

        gold = int(player.gold) if hasattr(player, 'gold') else 99

        # --- Phase 1: Sell excess from bench ---
        excess_bench = []
        for pos, c_id in enumerate(bench_state):
            if c_id is not None and _have(c_id) > _need(c_id):
                excess_bench.append((pos, c_id))
        for pos, c_id in excess_bench:
            if _have(c_id) <= _need(c_id):
                continue
            actions.append([3, pos + BOARD_SIZE, 0])
            owned_counts[c_id] = _have(c_id) - 1
            bench_state[pos] = None

        # --- Phase 2: Buy needed champions from shop ---
        for c_id in shop_ids:
            if c_id is None:
                continue
            if _have(c_id) >= _need(c_id):
                continue
            if c_id not in self.idx_to_name:
                continue
            champ_name = self.idx_to_name[c_id]
            cost = COST.get(champ_name, 0)
            if cost < 1 or gold < cost:
                continue
            if not any(s is None for s in bench_state):
                break
            actions.append([2, c_id, 0])
            gold -= cost
            for i in range(BENCH_SIZE):
                if bench_state[i] is None:
                    bench_state[i] = c_id
                    owned_counts[c_id] = _have(c_id) + 1
                    break

        # --- Phase 3: Sell excess from board (not needed anywhere) ---
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                current_c = int(current_board[y, x])
                target_c = int(target_board[y, x])
                if current_c == EMPTY_CLASS or target_c == current_c:
                    continue
                if _need(current_c) > 0 and _have(current_c) > _need(current_c):
                    pass
                elif _need(current_c) == 0:
                    pass
                else:
                    continue
                is_needed_elsewhere = False
                for ty in range(BOARD_HEIGHT):
                    for tx in range(BOARD_WIDTH):
                        if int(target_board[ty, tx]) == current_c and int(current_board[ty, tx]) != current_c:
                            is_needed_elsewhere = True
                            break
                    if is_needed_elsewhere:
                        break
                if is_needed_elsewhere:
                    continue
                if _have(current_c) <= _need(current_c):
                    continue
                dcord = y * BOARD_WIDTH + x
                vac = None
                for i in range(BENCH_SIZE):
                    if bench_state[i] is None:
                        vac = i
                        break
                if vac is not None:
                    actions.append([1, dcord, vac + BOARD_SIZE])
                    bench_state[vac] = current_c
                    current_board[y, x] = EMPTY_CLASS
                    owned_counts[current_c] = _have(current_c) - 1
                else:
                    actions.append([3, dcord, 0])
                    current_board[y, x] = EMPTY_CLASS
                    owned_counts[current_c] = _have(current_c) - 1

        # --- Phase 4: Place correct champions on target board positions ---
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                target_c = int(target_board[y, x])
                if target_c == EMPTY_CLASS:
                    continue
                current_c = int(current_board[y, x])
                if current_c == target_c:
                    continue
                src_is_bench = None
                src_pos = None
                for bp, bi in enumerate(bench_state):
                    if bi == target_c:
                        src_is_bench = True
                        src_pos = bp
                        break
                if src_is_bench is None:
                    for by in range(BOARD_HEIGHT):
                        for bx in range(BOARD_WIDTH):
                            if by == y and bx == x:
                                continue
                            if int(current_board[by, bx]) == target_c:
                                src_is_bench = False
                                src_pos = by * BOARD_WIDTH + bx
                                break
                        if src_is_bench is not None:
                            break
                if src_is_bench is None:
                    continue
                dcord = y * BOARD_WIDTH + x
                current_c = int(current_board[y, x])
                if src_is_bench:
                    actions.append([1, src_pos + BOARD_SIZE, dcord])
                    bench_state[src_pos] = None
                    if current_c != EMPTY_CLASS:
                        bench_state[src_pos] = current_c
                    current_board[y, x] = target_c
                else:
                    src_y, src_x = divmod(src_pos, BOARD_WIDTH)
                    actions.append([1, src_pos, dcord])
                    current_board[y, x] = target_c
                    if current_c != EMPTY_CLASS:
                        current_board[src_y, src_x] = current_c
                    else:
                        current_board[src_y, src_x] = EMPTY_CLASS

        # --- Phase 5: Clear remaining board positions the target does not want ---
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                current_c = int(current_board[y, x])
                if current_c == EMPTY_CLASS:
                    continue
                target_c = int(target_board[y, x])
                if target_c != EMPTY_CLASS:
                    continue
                is_needed_elsewhere = False
                for ty in range(BOARD_HEIGHT):
                    for tx in range(BOARD_WIDTH):
                        if int(target_board[ty, tx]) == current_c and int(current_board[ty, tx]) != current_c:
                            is_needed_elsewhere = True
                            break
                    if is_needed_elsewhere:
                        break
                if is_needed_elsewhere:
                    continue
                dcord = y * BOARD_WIDTH + x
                vac = None
                for i in range(BENCH_SIZE):
                    if bench_state[i] is None:
                        vac = i
                        break
                if vac is not None:
                    actions.append([1, dcord, vac + BOARD_SIZE])
                    bench_state[vac] = current_c
                    current_board[y, x] = EMPTY_CLASS
                else:
                    actions.append([3, dcord, 0])
                    current_board[y, x] = EMPTY_CLASS

        if not actions:
            actions.append([0, 0, 0])

        return actions

    def translate(self, board_probs: torch.Tensor, player, shop_slots: list = None) -> list:
        target_board = self.decode_target_board(board_probs)[0]

        current_board = self.get_current_board(player).copy()
        bench_state = [None] * BENCH_SIZE
        for pos, idx in self.get_current_bench(player):
            bench_state[pos] = idx

        actions = []

        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                target_c = int(target_board[y, x])
                current_c = int(current_board[y, x])

                if target_c == EMPTY_CLASS:
                    continue
                if current_c == target_c:
                    continue

                src_is_bench = None
                src_pos = None

                for bp, bi in enumerate(bench_state):
                    if bi == target_c:
                        src_is_bench = True
                        src_pos = bp
                        break

                if src_is_bench is None:
                    for by in range(BOARD_HEIGHT):
                        for bx in range(BOARD_WIDTH):
                            if by == y and bx == x:
                                continue
                            if int(current_board[by, bx]) == target_c:
                                src_is_bench = False
                                src_pos = by * BOARD_WIDTH + bx
                                break
                        if src_is_bench is not None:
                            break

                if src_is_bench is None:
                    continue

                dcord = y * BOARD_WIDTH + x
                current_c = int(current_board[y, x])

                if src_is_bench:
                    bench_dcord = src_pos + BOARD_SIZE
                    actions.append([1, bench_dcord, dcord])
                    bench_state[src_pos] = None
                    if current_c != EMPTY_CLASS:
                        bench_state[src_pos] = current_c
                    current_board[y, x] = target_c
                else:
                    src_y, src_x = divmod(src_pos, BOARD_WIDTH)
                    actions.append([1, src_pos, dcord])
                    current_board[y, x] = target_c
                    current_board[src_y, src_x] = current_c if current_c != EMPTY_CLASS else EMPTY_CLASS

        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                current_c = int(current_board[y, x])
                target_c = int(target_board[y, x])

                if current_c == EMPTY_CLASS:
                    continue
                if target_c != EMPTY_CLASS:
                    continue

                needed = False
                for ty in range(BOARD_HEIGHT):
                    for tx in range(BOARD_WIDTH):
                        if (int(target_board[ty, tx]) == current_c
                                and int(current_board[ty, tx]) != current_c):
                            needed = True
                            break
                    if needed:
                        break

                if needed:
                    continue

                dcord = y * BOARD_WIDTH + x
                bench_vacancy = None
                for i in range(BENCH_SIZE):
                    if bench_state[i] is None:
                        bench_vacancy = i
                        break

                if bench_vacancy is not None:
                    bench_dcord = bench_vacancy + BOARD_SIZE
                    actions.append([1, dcord, bench_dcord])
                    bench_state[bench_vacancy] = current_c
                    current_board[y, x] = EMPTY_CLASS
                else:
                    actions.append([3, dcord, 0])
                    current_board[y, x] = EMPTY_CLASS

        if not actions:
            actions.append([0, 0, 0])

        return actions

    def translate_batch(self, board_probs: torch.Tensor, players: list, shop_slots: list = None) -> list:
        batch_size = board_probs.shape[0]
        results = []
        for i in range(batch_size):
            if i < len(players):
                probs = board_probs[i : i + 1]
                results.append(self.translate(probs, players[i], shop_slots))
            else:
                results.append([[0, 0, 0]])
        return results
