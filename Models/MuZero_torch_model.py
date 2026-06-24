import torch
import config
import collections
import numpy as np
import time

# TFT-182: Embedding-based observation schema constants
# Embedding table sizes
NUM_CHAMPIONS = 58
CHAMPION_EMBED_DIM = 32
NUM_ITEMS = 37
ITEM_EMBED_DIM = 24
NUM_TRAITS = 20
TRAIT_EMBED_DIM = 8
NUM_ORIGINS = 10
ORIGIN_EMBED_DIM = 8

# Per-slot feature dimensions (post-embedding)
# champion(32) + items(3*24=72) + traits(8) + origins(8) + star(1) + chosen(1) = 122
PER_SLOT_DIM = 122
BOARD_SLOTS = 28
BENCH_CHAMP_SLOTS = 9
BENCH_ITEM_SLOTS = 10
SHOP_CHAMP_SLOTS = 5
NUM_OPPONENTS = 7

# Observation field dimensions
BOARD_DIM = BOARD_SLOTS * PER_SLOT_DIM  # 28 * 122 = 3416
BENCH_CHAMP_DIM = BENCH_CHAMP_SLOTS * PER_SLOT_DIM  # 9 * 122 = 1098
BENCH_ITEM_DIM = BENCH_ITEM_SLOTS * ITEM_EMBED_DIM  # 10 * 24 = 240
SHOP_CHAMP_DIM = SHOP_CHAMP_SLOTS * CHAMPION_EMBED_DIM  # 5 * 32 = 160
SHOP_CHOSEN_DIM = 1
TEAM_TRAITS_DIM = NUM_TRAITS  # 20
TEAM_ORIGINS_DIM = NUM_ORIGINS  # 10
PLAYER_STATE_DIM = 7
OPPONENT_BOARDS_DIM = NUM_OPPONENTS * BOARD_DIM  # 7 * 3416 = 23912
OPPONENT_INFO_DIM = NUM_OPPONENTS * 4  # 7 * 4 = 28

# Total observation size (without action mask)
TOTAL_OBS_DIM = (BOARD_DIM + BENCH_CHAMP_DIM + BENCH_ITEM_DIM +
                 SHOP_CHAMP_DIM + SHOP_CHOSEN_DIM +
                 TEAM_TRAITS_DIM + TEAM_ORIGINS_DIM +
                 PLAYER_STATE_DIM + OPPONENT_BOARDS_DIM +
                 OPPONENT_INFO_DIM)  # 28892

# Action mask size (7 selectors + 37 sources + 37 destinations)
ACTION_MASK_DIM = sum(config.ACTION_DIM)

# Full observation size including action mask
TOTAL_WITH_MASK = TOTAL_OBS_DIM + ACTION_MASK_DIM

NetworkOutput = collections.namedtuple(
    'NetworkOutput',
    'value reward policy_logits hidden_state')

def dcord_to_2dcord(dcord):
        x = dcord % 7
        y = (dcord - x) // 7
        return x, y

def action_to_3d(action):
    """Convert action to 3D format for TFTSet4Gym compatibility"""
    from Models.action_conversion import action_3d_to_policy
    action_2d = np.atleast_2d(action)
    batch_size = action_2d.shape[0]
    policy_size = config.ACTION_CONCAT_SIZE
    encoded = np.zeros((batch_size, 1, policy_size), dtype=np.float32)
    for i in range(batch_size):
        encoded[i, 0, :] = action_3d_to_policy(action_2d[i])
    return encoded

def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


class AbstractNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def initial_inference(self, observation):
        pass

    def recurrent_inference(self, encoded_state, action):
        pass

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)
        self.eval()

    # Renaming as to not override built-in functions
    def tft_save_model(self, episode):
        import os
        if not os.path.exists("./Checkpoints"):
            os.makedirs("./Checkpoints")

        path = f'./Checkpoints/checkpoint_{episode}'
        torch.save(self.state_dict(), path)

    # Renaming as to not override built-in functions
    def tft_load_model(self, episode):
        import os
        path = f'./Checkpoints/checkpoint_{episode}'
        if os.path.isfile(path):
            self.load_state_dict(torch.load(path))
            self.eval()
            print("Loading model episode {}".format(episode))
        else:
            print("Initializing model with new weights.")


class MuZeroNetwork(AbstractNetwork):
    def __init__(self):
        super().__init__()
        self.full_support_size = config.ENCODER_NUM_STEPS

        self.representation_network = RepNetwork(
            config.OBSERVATION_SIZE,
            [config.HIDDEN_STATE_SIZE] * 5,
            config.HIDDEN_STATE_SIZE,
            1
        ).cuda()

        # self.action_encodings = mlp(config.ACTION_CONCAT_SIZE, [config.LAYER_HIDDEN_SIZE] * 0,
        #                             config.HIDDEN_STATE_SIZE)

        self.dynamics_network = DynNetwork(
            config.HIDDEN_STATE_SIZE + config.ACTION_ENCODING_SIZE,
            [config.LAYER_HIDDEN_SIZE] * 6,
            config.HIDDEN_STATE_SIZE,
            self.full_support_size
        ).cuda()

        self.prediction_network = PredNetwork(
            config.HIDDEN_STATE_SIZE,
            [config.LAYER_HIDDEN_SIZE] * 6,
            1,
            self.full_support_size
        ).cuda()

    def prediction(self, encoded_state):
        policy_logits, value = self.prediction_network(encoded_state)
        return policy_logits, value

    def representation(self, observation):
        encoded_state = self.representation_network(observation)
        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_encoded_state = encoded_state.min(dim=1, keepdim=True)[0]
        max_encoded_state = encoded_state.max(dim=1, keepdim=True)[0]
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (
            encoded_state - min_encoded_state
        ) / scale_encoded_state
        return encoded_state_normalized

    def dynamics(self, hidden_state, action):
        # Convert action to numpy if it's a tensor
        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy()
        else:
            action_np = action

        cube_action = torch.from_numpy(action_to_3d(action_np)).float().to(hidden_state.device)

        next_hidden_state = self.dynamics_network(hidden_state, cube_action)

        # Synthesize dummy zero reward (terminal-only environment)
        reward = torch.zeros(next_hidden_state.shape[0], 1, device=next_hidden_state.device)

        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_hidden_state = next_hidden_state.min(dim=1, keepdim=True)[0]
        max_next_hidden_state = next_hidden_state.max(dim=1, keepdim=True)[0]
        scale_next_hidden_state = max_next_hidden_state - min_next_hidden_state
        scale_next_hidden_state[scale_next_hidden_state < 1e-5] += 1e-5
        next_hidden_state_normalized = (
            next_hidden_state - min_next_hidden_state
        ) / scale_next_hidden_state

        return next_hidden_state_normalized, reward

    def initial_inference(self, observation):
        device = next(self.parameters()).device
        if isinstance(observation, np.ndarray):
            observation_tensor = torch.from_numpy(observation).float().to(device)
        else:
            observation_tensor = observation.to(device)

        # Ensure observation_tensor size matches config.OBSERVATION_SIZE explicitly
        target_size = config.OBSERVATION_SIZE
        size = observation_tensor.shape[-1] if observation_tensor.ndim > 1 else observation_tensor.shape[0]
        if size != target_size:
            raise ValueError(f"Observation size {size} does not match config.OBSERVATION_SIZE {target_size}!")

        hidden_state = self.representation(observation_tensor)
        policy_logits, value_logits = self.prediction(hidden_state)

        reward = np.zeros(observation.shape[0])

        value = value_logits

        outputs = {
            "value": value,
            "reward": reward,
            "policy_logits": policy_logits,
            "hidden_state": hidden_state,
        }
        return outputs

    def recurrent_inference(self, hidden_state, action):
        next_hidden_state, reward_logits = self.dynamics(hidden_state, action)
        policy_logits, value_logits = self.prediction(next_hidden_state)

        # In evaluation mode (inference/MCTS), we detach to save memory and prevent graph leaks.
        if not self.training:
            return {
                "value": value_logits.detach(),
                "reward": reward_logits.detach(),
                "policy_logits": policy_logits.detach(),
                "hidden_state": next_hidden_state.detach()
            }

        return {
            "value": value_logits,
            "reward": reward_logits,
            "policy_logits": policy_logits,
            "hidden_state": next_hidden_state
        }

def mlp(input_size,
        layer_sizes,
        output_size,
        output_activation=torch.nn.Identity,
        activation=torch.nn.LeakyReLU):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers).cuda()


class PredNetwork(torch.nn.Module):
    def __init__(self, input_size, layer_sizes, output_size, encoding_size) -> torch.nn.Module:
        super().__init__()

        self.relu = torch.nn.LeakyReLU(inplace=True)
        self.sigmoid = torch.nn.Sigmoid()

        hidden = input_size
        layer_sizes = layer_sizes if layer_sizes else [hidden] * 6

        self.dense1 = torch.nn.Linear(hidden, layer_sizes[0])
        self.res_layers = torch.nn.ModuleList()
        self.res_lns = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.res_layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i]))
            self.res_lns.append(torch.nn.LayerNorm(layer_sizes[i]))

        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(layer_sizes[-1], layer_sizes[-1]),
            self.relu,
            torch.nn.Linear(layer_sizes[-1], 1)
        )
        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(layer_sizes[-1], layer_sizes[-1]),
            self.relu,
            torch.nn.Linear(layer_sizes[-1], config.ACTION_CONCAT_SIZE)
        )

    def forward(self, x):
        x = self.relu(self.dense1(x))
        for i in range(len(self.res_layers)):
            residual = x
            x = self.res_lns[i](self.res_layers[i](x))
            x = self.relu(x) + residual

        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value

    def __call__(self, x):
        return self.forward(x)


class RepNetwork(torch.nn.Module):
    """Representation network with learnable embeddings for TFT-182.

    Applies embedding lookups for champions, items, traits, and origins,
    then concatenates embedded features with scalar features before
    feeding into the residual MLP.
    """
    def __init__(self, input_size, layer_sizes, output_size, encoding_size) -> torch.nn.Module:
        super().__init__()
        hidden = output_size

        self.relu = torch.nn.LeakyReLU(inplace=True)

        # TFT-182: Learnable embedding tables
        # Champion embeddings: 58 champions -> 32-dim
        self.champion_embedding = torch.nn.Embedding(NUM_CHAMPIONS, CHAMPION_EMBED_DIM)

        # Item embeddings: 37 items -> 24-dim
        self.item_embedding = torch.nn.Embedding(NUM_ITEMS, ITEM_EMBED_DIM)

        # Trait embeddings: 20 traits -> 8-dim
        self.trait_embedding = torch.nn.Embedding(NUM_TRAITS, TRAIT_EMBED_DIM)

        # Origin embeddings: 10 origins -> 8-dim
        self.origin_embedding = torch.nn.Embedding(NUM_ORIGINS, ORIGIN_EMBED_DIM)

        # Initialize embeddings with small random values
        torch.nn.init.xavier_uniform_(self.champion_embedding.weight)
        torch.nn.init.xavier_uniform_(self.item_embedding.weight)
        torch.nn.init.xavier_uniform_(self.trait_embedding.weight)
        torch.nn.init.xavier_uniform_(self.origin_embedding.weight)

        # Compute total embedded feature dimension
        # Board: 28 slots * (champ(32) + 3*item(24) + traits(8) + origins(8) + star(1) + chosen(1))
        # = 28 * 122 = 3416
        # Bench champions: 9 * 122 = 1098
        # Bench items: 10 * 24 = 240
        # Shop champions: 5 * 32 = 160
        # Shop chosen: 1
        # Team traits: 20
        # Team origins: 10
        # Player state: 7
        # Opponent boards: 7 * 28 * 122 = 23912
        # Opponent info: 7 * 4 = 28
        # Total embedded dim for MLP input
        self.embedded_dim = (BOARD_DIM + BENCH_CHAMP_DIM + BENCH_ITEM_DIM +
                             SHOP_CHAMP_DIM + SHOP_CHOSEN_DIM +
                             TEAM_TRAITS_DIM + TEAM_ORIGINS_DIM +
                             PLAYER_STATE_DIM + OPPONENT_BOARDS_DIM +
                             OPPONENT_INFO_DIM)

        # First linear layer from embedded features to hidden state
        self.dense1 = torch.nn.Linear(self.embedded_dim, hidden)
        layer_sizes = layer_sizes if layer_sizes else [hidden] * 5
        self.res_layers = torch.nn.ModuleList()
        self.res_lns = torch.nn.ModuleList()
        for i, size in enumerate(layer_sizes):
            self.res_layers.append(torch.nn.Linear(hidden, size))
            self.res_lns.append(torch.nn.LayerNorm(size))

    def _extract_champion_id(self, slot_vector: torch.Tensor) -> torch.Tensor:
        """Extract champion ID from slot vector for a batch."""
        return torch.clamp(torch.round(slot_vector[:, 0]).long(), 0, NUM_CHAMPIONS - 1)

    def _extract_item_id(self, slot_vector: torch.Tensor, slot_idx: int) -> torch.Tensor:
        """Extract item ID from slot vector for a batch."""
        # item 0 at index 32, item 1 at 56, item 2 at 80
        item_val = slot_vector[:, 32 + (slot_idx * 24)]
        return torch.clamp(torch.round(item_val).long(), 0, NUM_ITEMS - 1)

    def _extract_trait_id(self, slot_vector: torch.Tensor, trait_idx: int) -> torch.Tensor:
        """Extract trait ID from slot vector for a batch."""
        # trait 0 at index 104, trait_idx == 1 is not used under 122-dim layout
        if trait_idx == 0:
            val = slot_vector[:, 104]
        else:
            val = torch.zeros_like(slot_vector[:, 104])
        return torch.clamp(torch.round(val).long(), 0, NUM_TRAITS - 1)

    def _extract_origin_id(self, slot_vector: torch.Tensor, origin_idx: int) -> torch.Tensor:
        """Extract origin ID from slot vector for a batch."""
        # origin 0 at index 112, origin_idx == 1 is not used under 122-dim layout
        if origin_idx == 0:
            val = slot_vector[:, 112]
        else:
            val = torch.zeros_like(slot_vector[:, 112])
        return torch.clamp(torch.round(val).long(), 0, NUM_ORIGINS - 1)

    def _encode_slot(self, slot_vector: torch.Tensor) -> torch.Tensor:
        """Apply embedding lookups for a single board/bench slot across a batch."""
        # Mask of active slots: shape (batch, 1)
        active_mask = (slot_vector.abs().sum(dim=1, keepdim=True) >= 1e-6).float()

        # Get champion ID and apply embedding
        champ_id = self._extract_champion_id(slot_vector)
        champ_embed = self.champion_embedding(champ_id) * active_mask  # (batch, 32)

        # Get item IDs and apply embeddings
        item_embeds = []
        for i in range(3):
            item_id = self._extract_item_id(slot_vector, i)
            item_embeds.append(self.item_embedding(item_id))  # each (batch, 24)
        three_items = torch.cat(item_embeds, dim=1) * active_mask  # (batch, 72)

        # Get trait IDs and apply embeddings
        trait_embeds = []
        for i in range(2):
            trait_id = self._extract_trait_id(slot_vector, i)
            trait_embeds.append(self.trait_embedding(trait_id))  # each (batch, 8)
        traits_embed = (trait_embeds[0] + trait_embeds[1]) * active_mask  # (batch, 8)

        # Get origin IDs and apply embeddings
        origin_embeds = []
        for i in range(2):
            origin_id = self._extract_origin_id(slot_vector, i)
            origin_embeds.append(self.origin_embedding(origin_id))  # each (batch, 8)
        origins_embed = (origin_embeds[0] + origin_embeds[1]) * active_mask  # (batch, 8)

        # Concatenate: champ(32) + items(72) + traits(8) + origins(8) + star(1) + chosen(1) = 122
        star_level = slot_vector[:, 120:121] * active_mask
        chosen_flag = slot_vector[:, 121:122] * active_mask
        slot_embed = torch.cat([champ_embed, three_items, traits_embed,
                                origins_embed, star_level, chosen_flag], dim=1)
        return slot_embed

    def forward(self, x):
        # Flatten input to [batch, total_obs_dim]
        x = torch.flatten(x, start_dim=1)
        device = x.device

        # TFT-182: Apply embedding lookups and build embedded representation
        offset = 0

        # 1. Board: (batch, 28, 122) -> (batch, 28, 122) embedded
        board_shape = x[:, offset:offset + BOARD_DIM].view(-1, BOARD_SLOTS, PER_SLOT_DIM)
        offset += BOARD_DIM
        board_embedded = []
        for slot in range(BOARD_SLOTS):
            slot_vec = board_shape[:, slot, :]  # (batch, 122)
            slot_embed = self._encode_slot(slot_vec)  # (batch, 122)
            board_embedded.append(slot_embed)
        board_repr = torch.cat(board_embedded, dim=1)  # (batch, 28*122=3416)

        # 2. Bench champions: (batch, 9, 122) -> (batch, 9*122=1098) embedded
        bench_champ_shape = x[:, offset:offset + BENCH_CHAMP_DIM].view(
            -1, BENCH_CHAMP_SLOTS, PER_SLOT_DIM)
        offset += BENCH_CHAMP_DIM
        bench_champ_embedded = []
        for slot in range(BENCH_CHAMP_SLOTS):
            slot_vec = bench_champ_shape[:, slot, :]
            slot_embed = self._encode_slot(slot_vec)
            bench_champ_embedded.append(slot_embed)
        bench_champ_repr = torch.cat(bench_champ_embedded, dim=1)

        # 3. Bench items: (batch, 10, 24) - item indices in first dim, embed
        bench_item_shape = x[:, offset:offset + BENCH_ITEM_DIM].view(
            -1, BENCH_ITEM_SLOTS, ITEM_EMBED_DIM)
        offset += BENCH_ITEM_DIM
        bench_item_embeds = []
        for slot in range(BENCH_ITEM_SLOTS):
            slot_vec = bench_item_shape[:, slot, :]  # (batch, 24)
            active_mask = (slot_vec.abs().sum(dim=1, keepdim=True) >= 1e-6).float()
            item_id = torch.clamp(torch.round(slot_vec[:, 0]).long(), 0, NUM_ITEMS - 1)
            item_embed = self.item_embedding(item_id) * active_mask  # (batch, 24)
            bench_item_embeds.append(item_embed)
        bench_items_repr = torch.cat(bench_item_embeds, dim=1)  # (batch, 10*24=240)

        # 4. Shop champions: (batch, 5, 32) - champion indices in first dim, embed
        shop_shape = x[:, offset:offset + SHOP_CHAMP_DIM].view(
            -1, SHOP_CHAMP_SLOTS, CHAMPION_EMBED_DIM)
        offset += SHOP_CHAMP_DIM
        shop_champ_embeds = []
        for slot in range(SHOP_CHAMP_SLOTS):
            slot_vec = shop_shape[:, slot, :]  # (batch, 32)
            active_mask = (slot_vec.abs().sum(dim=1, keepdim=True) >= 1e-6).float()
            champ_id = torch.clamp(torch.round(slot_vec[:, 0]).long(), 0, NUM_CHAMPIONS - 1)
            champ_embed = self.champion_embedding(champ_id) * active_mask  # (batch, 32)
            shop_champ_embeds.append(champ_embed)
        shop_champs_repr = torch.cat(shop_champ_embeds, dim=1)  # (batch, 5*32=160)

        # 5. Shop chosen: scalar
        shop_chosen = x[:, offset:offset + SHOP_CHOSEN_DIM]
        offset += SHOP_CHOSEN_DIM

        # 6. Team traits: (batch, 20) - normalized floats
        team_traits = x[:, offset:offset + TEAM_TRAITS_DIM]
        offset += TEAM_TRAITS_DIM

        # 7. Team origins: (batch, 10) - normalized floats
        team_origins = x[:, offset:offset + TEAM_ORIGINS_DIM]
        offset += TEAM_ORIGINS_DIM

        # 8. Player state: (batch, 7) - scalars
        player_state = x[:, offset:offset + PLAYER_STATE_DIM]
        offset += PLAYER_STATE_DIM

        # 9. Opponent boards: (batch, 7, 28, 122) -> (batch, 7*28*122=23912) embedded
        opp_boards_shape = x[:, offset:offset + OPPONENT_BOARDS_DIM].view(
            -1, NUM_OPPONENTS, BOARD_SLOTS, PER_SLOT_DIM)
        offset += OPPONENT_BOARDS_DIM
        opp_boards_list = []
        for opp in range(NUM_OPPONENTS):
            opp_board = opp_boards_shape[:, opp, :, :]  # (batch, 28, 122)
            opp_embedded = []
            for slot in range(BOARD_SLOTS):
                slot_vec = opp_board[:, slot, :]
                slot_embed = self._encode_slot(slot_vec)
                opp_embedded.append(slot_embed)
            opp_board_repr = torch.cat(opp_embedded, dim=1)
            opp_boards_list.append(opp_board_repr)
        opp_boards_repr = torch.cat(opp_boards_list, dim=1)

        # 10. Opponent info: (batch, 7, 4) - scalars
        opp_info = x[:, offset:offset + OPPONENT_INFO_DIM]
        offset += OPPONENT_INFO_DIM

        # Concatenate all embedded features
        embedded = torch.cat([
            board_repr,
            bench_champ_repr,
            bench_items_repr,
            shop_champs_repr,
            shop_chosen,
            team_traits,
            team_origins,
            player_state,
            opp_boards_repr,
            opp_info
        ], dim=1)  # (batch, embedded_dim)

        # Feed through residual MLP
        x = self.relu(self.dense1(embedded))
        for i in range(len(self.res_layers)):
            residual = x
            x = self.res_lns[i](self.res_layers[i](x))
            x = self.relu(x) + residual

        return x

    def __call__(self, x):
        return self.forward(x)


class DynNetwork(torch.nn.Module):
    def __init__(self, input_size, layer_sizes, output_size, encoding_size) -> torch.nn.Module:
        super().__init__()
        self.relu = torch.nn.LeakyReLU(inplace=True)
        layer_sizes = layer_sizes if layer_sizes else [input_size] * 6
        layer_sizes = list(layer_sizes)
        hidden = config.HIDDEN_STATE_SIZE
        if len(layer_sizes) > 0:
            layer_sizes = [hidden] * len(layer_sizes)
        self.dense1 = torch.nn.Linear(input_size, hidden)
        for i, size in enumerate(layer_sizes):
            prev_size = layer_sizes[i - 1] if i > 0 else layer_sizes[0]
            setattr(self, f'dense{i + 2}', torch.nn.Linear(prev_size, size))
        self.layer_sizes = layer_sizes

    def forward(self, x, action):
        action = torch.squeeze(action, dim=1)
        x = torch.cat((x, action), dim=1)
        x = self.relu(self.dense1(x))
        for i in range(len(self.layer_sizes)):
            layer = getattr(self, f'dense{i + 2}')
            residual = x
            x = self.relu(layer(x))
            x = x + residual
        return x

    def __call__(self, x, action):
        return self.forward(x, action)

def resnet(input_size,
        layer_sizes,
        output_size):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(0, len(sizes) - 1):
        layers += [ResLayer(sizes[i], sizes[i + 1])]

    return torch.nn.Sequential(*layers).cuda()

class MultiMlp(torch.nn.Module):
    def __init__(self,
                 input_size,
                 layer_sizes,
                 output_sizes,
                 output_activation=torch.nn.Identity,
                 activation=torch.nn.LeakyReLU):
        super().__init__()

        sizes = [input_size] + layer_sizes
        layers = []
        for i in range(len(sizes) - 1):
            act = activation
            layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
        self.encoding_layer = torch.nn.Sequential(*layers).cuda()

        self.head_0 = torch.nn.Sequential(
                torch.nn.Linear(layer_sizes[-1], output_sizes[0])
            ).cuda()

    def forward(self, x):
        x = self.encoding_layer(x)
        output = []
        output.append(self.head_0(x))
        return output

    def __call__(self, x):
        return self.forward(x)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> torch.nn.Conv2d:
    """1x1 convolution"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResLayer(torch.nn.Module):
    def __init__(self, input_channels, n_kernels) -> torch.nn.Module:
        super().__init__()

        self.conv1 = torch.nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.conv2 = torch.nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.relu = torch.nn.LeakyReLU(inplace=True)

    def forward(self, x):
        input = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += input

        return self.relu(out)

    def __call__(self, x):
        return self.forward(x)

class ValueEncoder:
    """Encoder for reward and value targets from Appendix of MuZero Paper."""

    def __init__(self,
                 min_value,
                 max_value,
                 num_steps,
                 use_contractive_mapping=True):
        if not max_value > min_value:
            raise ValueError('max_value must be > min_value')
        min_value = float(min_value)
        max_value = float(max_value)
        if use_contractive_mapping:
            max_value = contractive_mapping(max_value)
            min_value = contractive_mapping(min_value)
        if num_steps <= 0:
            num_steps = np.ceil(max_value) + 1 - np.floor(min_value)
        self.min_value = min_value
        self.max_value = max_value
        self.value_range = max_value - min_value
        self.num_steps = num_steps
        self.step_size = self.value_range / (num_steps - 1)
        self.step_range_int = np.arange(0, self.num_steps, dtype=int)
        self.step_range_float = self.step_range_int.astype(float)
        self.use_contractive_mapping = use_contractive_mapping

    def encode(self, value):  # not worth optimizing
        if len(value.shape) != 1:
            raise ValueError(
                'Expected value to be 1D Tensor [batch_size], but got {}.'.format(
                    value.shape))
        if self.use_contractive_mapping:
            value = contractive_mapping(value)
        value = np.expand_dims(value, -1)
        clipped_value = np.clip(value, self.min_value, self.max_value)
        above_min = clipped_value - self.min_value
        num_steps = above_min / self.step_size
        lower_step = np.floor(num_steps)
        upper_mod = num_steps - lower_step
        lower_step = lower_step.astype(int)
        upper_step = lower_step + 1
        lower_mod = 1.0 - upper_mod
        lower_encoding, upper_encoding = (
            np.equal(step, self.step_range_int).astype(float) * mod
            for step, mod in (
                (lower_step, lower_mod),
                (upper_step, upper_mod),)
        )
        return lower_encoding + upper_encoding

    def decode(self, logits):  # not worth optimizing
        if len(logits.shape) != 2:
            raise ValueError('Expected logits to be 2D Tensor [batch_size, steps], but got {}.'
                .format(logits.shape))
        num_steps = np.sum(logits * self.step_range_float, -1)
        above_min = num_steps * self.step_size
        value = above_min + self.min_value
        if self.use_contractive_mapping:
            value = inverse_contractive_mapping(value)
        return value


# From the MuZero paper.
def contractive_mapping(x, eps=0.001):
    return np.sign(x) * (np.sqrt(np.abs(x) + 1.) - 1.) + eps * x


# From the MuZero paper.
def inverse_contractive_mapping(x, eps=0.001):
    return np.sign(x) * \
           (np.square((np.sqrt(4 * eps * (np.abs(x) + 1. + eps) + 1.) - 1.) / (2. * eps)) - 1.)

# Softmax function in np because we're converting it anyway
def softmax_stable(x):
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()
