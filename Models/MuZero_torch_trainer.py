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
        # Updated for TFTSet4Gym: policy shape is (batch, time_steps, 3, 37) 
        # Flatten last two dims: 3 * 37 = 111
        target_policy = torch.reshape(torch.tensor(np.array(target_policy)), (-1, num_target_steps, config.ACTION_CONCAT_SIZE)).to(device)
        
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
            policy_log_probs = torch.nn.functional.log_softmax(policy_logits_flat, dim=-1)
            policy_loss += kl_loss_fn(policy_log_probs, target_policy_normalized)

            accs['value_diff'].append(torch.abs(torch.squeeze(value) - target_value[:, tstep]))

            accs['value'].append(torch.squeeze(value))
            accs['policy'].append(torch.squeeze(policy_logits_flat))

            accs['target_value'].append(target_value[:, tstep])
            accs['target_policy'].append(target_policy[:, tstep])

        accs = {k: torch.stack(v, -1) for k, v in accs.items()}

        mean_loss = value_loss.mean() + policy_loss.mean()
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
            summary_writer.add_scalar('losses/total', torch.mean(mean_loss), train_step)

            summary_writer.add_scalar('accuracy/value', -get_mean('value_diff'), train_step)
            summary_writer.flush()

        return mean_loss
