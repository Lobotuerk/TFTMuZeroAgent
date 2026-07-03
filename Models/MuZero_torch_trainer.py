import config
import collections
import torch
import numpy as np


Prediction = collections.namedtuple(
    'Prediction',
    'value reward policy_logits')


class Trainer(object):
    def __init__(self):
        self.optimizer = None
        self.scheduler = None

    def create_optimizer(self, agent):
        optimizer = torch.optim.Adam(agent.parameters(), lr=config.INIT_LEARNING_RATE)
        return optimizer

    def create_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.LEARNING_RATE_DECAY,
            eta_min=config.INIT_LEARNING_RATE * config.LR_DECAY_FUNCTION
        )

    def train_network(self, batch, combats, agent, train_step, summary_writer):
        if self.optimizer is None:
            self.optimizer = self.create_optimizer(agent)
            self.scheduler = self.create_scheduler(self.optimizer)
        observation, action, value, reward, policy, target_obs, bootstrap_depth = batch
        agent.train()

        self.optimizer.zero_grad()

        loss = self.compute_loss(agent, observation, action, value, reward, policy, target_obs, bootstrap_depth, combats, train_step, summary_writer)

        loss = loss.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=config.MAX_GRAD_NORM)

        self.optimizer.step()
        self.scheduler.step()

    def compute_loss(self, agent, observation, action, target_value, target_reward, target_policy, target_obs, bootstrap_depth, combats, train_step, summary_writer):

        device = next(agent.parameters()).device
        target_value = torch.from_numpy(target_value).float().to(device)
        bootstrap_depth = torch.from_numpy(bootstrap_depth).float().to(device)

        # initial step
        output = agent.initial_inference(observation)

        predictions = [
            Prediction(
                value=output["value"],
                reward=output["reward"],
                policy_logits=output["policy_logits"],
            )
        ]

        # recurrent steps
        num_recurrent_steps = config.UNROLL_STEPS-1
        for rstep in range(num_recurrent_steps):
            # 1.0 and 0.5 because we scale prior to the recurrent inference. 0.5 if scaled afterwards.
            hidden_state_gradient_scale = 1.0 if rstep == 0 else 0.5
            hidden_state = output["hidden_state"]
            hidden_state.requires_grad_(True).register_hook(lambda grad: grad * hidden_state_gradient_scale)
            output = agent.recurrent_inference(
                hidden_state,
                action[:, rstep],
            )
            predictions.append(
                Prediction(
                    value=output["value"],
                    reward=output["reward"],
                    policy_logits=output["policy_logits"],
                ))

        num_target_steps = target_value.shape[-1]
        assert len(predictions) == num_target_steps, (
            'There should be as many predictions ({}) as targets ({})'.format(
                len(predictions), num_target_steps))

        accs = collections.defaultdict(list)
        # Policy is a concatenation of 3 independent blocks per ACTION_DIM
        target_policy = torch.reshape(torch.tensor(np.array(target_policy)), (-1, num_target_steps, config.ACTION_CONCAT_SIZE)).to(device)

        # Compute n-step bootstrap targets via backward accumulation
        # Flatten target_obs for batched inference
        if target_obs[0] is not None:
            target_obs_array = np.array([t if t is not None else target_obs[0] for t in target_obs])
            with torch.no_grad():
                target_output = agent.initial_inference(target_obs_array)
                v_t_plus_n = target_output["value"]

            # Convert target_reward to tensor on correct device
            target_reward = torch.from_numpy(target_reward).float().to(device)

            discount = torch.tensor(config.DISCOUNT, device=device)
            num_steps = target_value.shape[1]
            new_target_value = torch.zeros_like(target_value)

            # Initialize current_z with bootstrap value
            current_z = (discount ** bootstrap_depth) * v_t_plus_n.squeeze(-1)

            # Backward accumulation through unroll steps
            for t in reversed(range(num_steps)):
                current_z = target_reward[:, t] + discount * current_z
                new_target_value[:, t] = current_z
            target_value = new_target_value

        # Precompute split indices from ACTION_DIM
        dims = list(config.ACTION_DIM)

        # Initialize losses as tensors with proper shape
        batch_size = target_value.shape[0]
        value_loss = torch.zeros(batch_size, device=device)
        policy_loss = torch.tensor(0.0, device=device)

        # Define loss functions
        MSE_loss = torch.nn.MSELoss(reduction='none')
        kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
        for tstep, prediction in enumerate(predictions):
            value = prediction.value
            policy_logits = prediction.policy_logits

            value_loss_step = MSE_loss(value.squeeze(), target_value[:, tstep])
            value_loss += value_loss_step

           # Split logits and targets into 3 independent blocks
            logits_flat = policy_logits.view(policy_logits.shape[0], -1)
            logits_blocks = torch.split(logits_flat, dims, dim=-1)
            target_blocks = torch.split(target_policy[:, tstep], dims, dim=-1)

            # Compute per-block KL divergence and sum them
            block_kl = []
            for block_logits, block_target in zip(logits_blocks, target_blocks):
                log_probs = torch.nn.functional.log_softmax(block_logits, dim=-1)
                block_kl.append(kl_loss_fn(log_probs, block_target))
            policy_loss += sum(block_kl)

            accs['value_diff'].append(torch.abs(torch.squeeze(value) - target_value[:, tstep]))

            accs['value'].append(torch.squeeze(value))
            accs['policy'].append(torch.squeeze(logits_flat))

            accs['target_value'].append(target_value[:, tstep])
            accs['target_policy'].append(target_policy[:, tstep])

        if len(combats) > 0:
            obs, results = combats
            if obs.ndim == 2:
                obs_flat = obs
            elif obs.ndim == 4:
                obs_flat = obs.reshape(obs.shape[0], -1)
            else:
                obs_flat = obs

            # Ensure obs_flat has correct size for initial_inference
            target_size = config.OBSERVATION_SIZE
            if obs_flat.shape[1] != target_size:
                raise ValueError(f"Combat observation size {obs_flat.shape[1]} does not match config.OBSERVATION_SIZE {target_size}!")
            
            output = agent.initial_inference(obs_flat)
            hidden_states = output["hidden_state"]
            hidden_flat = hidden_states.view(hidden_states.size(0), -1)
            
            torch_results = torch.from_numpy(results).float().to(device)
            torch_results = torch_results.view(-1)
            
            triplet_candidates = []
            for i in range(len(torch_results)):
                anchor_label = torch_results[i]
                pos_indices = (torch_results == anchor_label).nonzero(as_tuple=True)[0]
                neg_indices = (torch_results != anchor_label).nonzero(as_tuple=True)[0]
                
                pos_indices = pos_indices[pos_indices != i]
                if len(pos_indices) > 0 and len(neg_indices) > 0:
                    for p in pos_indices:
                        for n in neg_indices:
                            triplet_candidates.append((i, p.item(), n.item()))
            
            if len(triplet_candidates) > 0:
                # Limit the number of triplets to prevent combinatorial memory explosion / CUDA OOM
                max_triplets = 3000
                if len(triplet_candidates) > max_triplets:
                    import random
                    # Seed random for deterministic execution within a single step if needed, or keep it stochastic
                    sampled_indices = random.sample(range(len(triplet_candidates)), max_triplets)
                    triplets = [triplet_candidates[idx] for idx in sampled_indices]
                else:
                    triplets = triplet_candidates

                anchors = torch.stack([hidden_flat[t[0]] for t in triplets])
                positives = torch.stack([hidden_flat[t[1]] for t in triplets])
                negatives = torch.stack([hidden_flat[t[2]] for t in triplets])
                
                triplet_loss_fn = torch.nn.TripletMarginLoss(margin=1.0, p=2)
                combat_board_loss = triplet_loss_fn(anchors, positives, negatives)
            else:
                combat_board_loss = torch.tensor(0.0, device=device, requires_grad=True)

        accs = {k: torch.stack(v, -1) for k, v in accs.items()}

        mean_loss = value_loss.mean() + policy_loss.mean()
        if len(combats) > 0:
            mean_loss += combat_board_loss.mean()
        mean_loss.register_hook(lambda grad: grad * (1 / config.UNROLL_STEPS))

        sum_accs = {k: torch.sum(a, -1) for k, a in accs.items()}

        def get_mean(k):
            return torch.mean(sum_accs[k])

        if summary_writer is not None:
            summary_writer.add_scalar('prediction/value', get_mean('value'), train_step)
            summary_writer.add_scalar('prediction/value_variance', torch.mean(torch.var(accs['value'], dim=0)), train_step)
            summary_writer.add_scalar('prediction/policy_variance', torch.mean(torch.var(accs['policy'], dim=1)), train_step)

            summary_writer.add_scalar('target/value', get_mean('target_value'), train_step)
            summary_writer.add_scalar('target/value_variance', torch.mean(torch.var(accs['target_value'], dim=0)), train_step)
            summary_writer.add_scalar('target/policy_variance', torch.mean(torch.var(accs['target_policy'], dim=0)), train_step)

            summary_writer.add_scalar('losses/value', torch.mean(value_loss), train_step)
            summary_writer.add_scalar('losses/policy', torch.mean(policy_loss), train_step)
            if len(combats) > 0:
                summary_writer.add_scalar('losses/combat_contrastive', combat_board_loss.mean(), train_step)
            summary_writer.add_scalar('losses/total', torch.mean(mean_loss), train_step)

            summary_writer.add_scalar('accuracy/value', -get_mean('value_diff'), train_step)

            # Policy entropy: H(p) = -sum(p * log(p))
            policy_entropies = []
            for tstep, prediction in enumerate(predictions):
                logits = prediction.policy_logits
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log(probs + 1e-10)
                entropy = -(probs * log_probs).sum(dim=-1).mean()
                policy_entropies.append(entropy)
            summary_writer.add_scalar('metrics/policy_entropy', torch.mean(torch.stack(policy_entropies)), train_step)

            # Value regression error (MAE)
            value_mae = torch.mean(torch.abs(torch.squeeze(output["value"]) - target_value[:, 0]))
            summary_writer.add_scalar('metrics/value_mae', value_mae, train_step)

            summary_writer.flush()

        return mean_loss
