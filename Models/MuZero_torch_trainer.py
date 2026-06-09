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
        observation, action, value, reward, policy = batch
        agent.train()

        self.optimizer.zero_grad()

        loss = self.compute_loss(agent, observation, action, value, reward, policy, combats, train_step, summary_writer)

        loss = loss.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=config.MAX_GRAD_NORM)

        self.optimizer.step()
        self.scheduler.step()

    def compute_loss(self, agent, observation, action, target_value, target_reward, target_policy, combats, train_step, summary_writer):

        device = next(agent.parameters()).device
        target_value = torch.from_numpy(target_value).to(device)

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
        target_policy = torch.reshape(torch.tensor(np.array(target_policy)), (-1, num_target_steps, config.ACTION_CONCAT_SIZE)).to(device)

        dim_sizes = config.ACTION_DIM
        block_offsets = [0] + list(np.cumsum(dim_sizes))

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

            policy_logits_flat = policy_logits.view(policy_logits.shape[0], -1)

            target_policy_normalized = target_policy[:, tstep]

            step_policy_loss = torch.tensor(0.0, device=device)
            for i in range(len(dim_sizes)):
                s = block_offsets[i]
                e = block_offsets[i + 1]
                block_logits = policy_logits_flat[:, s:e]
                block_target = target_policy_normalized[:, s:e]
                block_log_probs = torch.nn.functional.log_softmax(block_logits, dim=-1)
                step_policy_loss = step_policy_loss + kl_loss_fn(block_log_probs, block_target)
            policy_loss = policy_loss + step_policy_loss

            accs['value_diff'].append(torch.abs(torch.squeeze(value) - target_value[:, tstep]))

            accs['value'].append(torch.squeeze(value))
            accs['policy'].append(torch.squeeze(policy_logits_flat))

            accs['target_value'].append(target_value[:, tstep])
            accs['target_policy'].append(target_policy[:, tstep])

        if len(combats) > 0:
            obs, results = combats
            # Ensure obs is 4D for spatial loss calculation [batch, depth, height, width]
            if obs.ndim == 2:
                # If flat, reshape to standard TFT observation shape (184, 4, 7)
                obs_4d = obs.reshape(obs.shape[0], 184, 4, 7)
                obs_flat = obs
            elif obs.ndim == 4:
                obs_4d = obs
                obs_flat = obs.reshape(obs.shape[0], -1)
            else:
                # Fallback for unexpected shapes
                obs_4d = obs
                obs_flat = obs

            # Ensure obs_flat has correct size for initial_inference
            target_size = config.OBSERVATION_SIZE
            if obs_flat.shape[1] < target_size:
                obs_flat = np.pad(obs_flat, ((0, 0), (0, target_size - obs_flat.shape[1])))
            elif obs_flat.shape[1] > target_size:
                obs_flat = obs_flat[:, :target_size]
            
            output = agent.initial_inference(obs_flat)
            hidden_states = output["hidden_state"]
            hidden_flat = hidden_states.view(hidden_states.size(0), -1)
            
            torch_results = torch.from_numpy(results).float().to(device)
            torch_results = torch_results.view(-1)
            
            anchors, positives, negatives = [], [], []
            for i in range(len(torch_results)):
                anchor_label = torch_results[i]
                pos_indices = (torch_results == anchor_label).nonzero(as_tuple=True)[0]
                neg_indices = (torch_results != anchor_label).nonzero(as_tuple=True)[0]
                
                pos_indices = pos_indices[pos_indices != i]
                if len(pos_indices) > 0 and len(neg_indices) > 0:
                    for p in pos_indices:
                        for n in neg_indices:
                            anchors.append(hidden_flat[i])
                            positives.append(hidden_flat[p])
                            negatives.append(hidden_flat[n])
            
            if len(anchors) > 0:
                anchors = torch.stack(anchors)
                positives = torch.stack(positives)
                negatives = torch.stack(negatives)
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
            summary_writer.add_scalar('target/policy_variance', torch.mean(torch.var(accs['target_policy'], dim=1)), train_step)

            summary_writer.add_scalar('losses/value', torch.mean(value_loss), train_step)
            summary_writer.add_scalar('losses/policy', torch.mean(policy_loss), train_step)
            if len(combats) > 0:
                summary_writer.add_scalar('losses/combat_contrastive', combat_board_loss.mean(), train_step)
            summary_writer.add_scalar('losses/total', torch.mean(mean_loss), train_step)

            summary_writer.add_scalar('accuracy/value', -get_mean('value_diff'), train_step)
            summary_writer.flush()

        return mean_loss
