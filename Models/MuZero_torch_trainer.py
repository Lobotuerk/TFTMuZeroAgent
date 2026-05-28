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

        # sample_set, policy = split_batch(sample_set, policy)  # [unroll_steps, num_dims, [(batch_size, dim) ...] ]

        loss = self.compute_loss(agent, observation, action, value, reward, policy, combats, train_step, summary_writer)

        loss = loss.mean()

        loss.backward()

        self.optimizer.step()
        self.scheduler.step()

    def compute_loss(self, agent, observation, action, target_value, target_reward, target_policy, combats, train_step, summary_writer):

        device = next(agent.parameters()).device
        target_reward = torch.from_numpy(target_reward).to(device)
        target_value = torch.from_numpy(target_value).to(device)

        # initial step
        output, directive, board_distribution = agent.initial_inference(observation)

        predictions = [
            Prediction(
                value=output["value"],
                reward=output["reward"],
                policy_logits=output["policy_logits"],
                # directive_value=output_directive["value"],
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
                    # directive_value=output_directive["value"],
                ))

        num_target_steps = target_value.shape[-1]
        assert len(predictions) == num_target_steps, (
            'There should be as many predictions ({}) as targets ({})'.format(
                len(predictions), num_target_steps))

        # target_reward_encoded, target_value_encoded = (torch.reshape(
        #     torch.tensor(enc.encode(torch.reshape(v, (-1,)).to('cpu'))).to('cuda'),
        #     (-1, num_target_steps,
        #      int(enc.num_steps))) for enc, v in ((agent.reward_encoder, target_reward),
        #                                          (agent.value_encoder, target_value)))

        accs = collections.defaultdict(list)
        # Updated for TFTSet4Gym: policy shape is (batch, time_steps, 3, 37) 
        # Flatten last two dims: 3 * 37 = 111
        target_policy = torch.reshape(torch.tensor(np.array(target_policy)), (-1, num_target_steps, 111)).to(device)
        
        # Initialize losses as tensors with proper shape
        batch_size = target_value.shape[0]
        value_loss = torch.zeros(batch_size, device=device)
        policy_loss = torch.tensor(0.0, device=device)
        directive_loss = torch.zeros(batch_size, device=device)

        # Define loss functions
        MAE_loss = torch.nn.L1Loss(reduction='none')
        cross_loss = torch.nn.CrossEntropyLoss(reduction='none')
        kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
        for tstep, prediction in enumerate(predictions):
            # prediction.value_logits is [batch_size, 601]

            # TODO: Possibly keep them as tensors in the inference functions
            value = prediction.value
            # print("VALUE", max(value).item(), min(value).item(), (max(value) - min(value)).item())
            # reward = prediction.reward
            # value_logits = prediction.value_logits
            # reward_logits = prediction.reward_logits.to('cuda') if torch.is_tensor(prediction.reward_logits) \
            #     else torch.tensor(prediction.reward_logits).to('cuda')
            # reward_logits = reward_logits.requires_grad_(True)
            policy_logits = prediction.policy_logits

            value_loss_step = MAE_loss(value.squeeze(), target_value[:, tstep])
            value_loss += value_loss_step
            # value_loss = (-target_value_encoded[:, tstep] *
            #               torch.nn.LogSoftmax(dim=-1)(value_logits)).sum(-1)
            # value_loss.register_hook(lambda grad: grad / config.UNROLL_STEPS)

            # accs['value_loss'].append(
            #     value_loss
            # )

            # reward_loss = (-target_reward_encoded[:, tstep] *
            #                torch.nn.LogSoftmax(dim=-1)(reward_logits)).sum(-1)
            # reward_loss.register_hook(lambda grad: grad / config.UNROLL_STEPS)

            # accs['reward_loss'].append(
            #     reward_loss
            # )

            # future ticket
            # entropy_loss = -tfd.Independent(tfd.Categorical(
            #     logits = logits, dtype=float), reinterpreted_batch_ndims=1).entropy()
            #     * config.policy_loss_entropy_regularizer

            # predictions.policy_logits is (batch_size, 3, 37) from TFTSet4Gym model
            # target_policy is (batch_size, unroll_steps, 111) - flattened policy
            
            # Flatten policy_logits to match target_policy shape: (batch_size, 3*37=111)
            policy_logits_flat = policy_logits.view(policy_logits.shape[0], -1)
            
            target_policy_normalized = torch.nn.functional.softmax(target_policy[:, tstep], dim=-1)
            policy_log_probs = torch.nn.functional.log_softmax(policy_logits_flat, dim=-1)
            policy_loss += kl_loss_fn(policy_log_probs, target_policy_normalized)
            # policy_loss = []
            # for batch_idx in range(len(target_policy[tstep])):
            #     local_policy_loss = (-torch.tensor(target_policy[tstep][batch_idx]).cuda() *
            #                                   torch.log(torch.tensor(policy_logits[0][batch_idx]).cuda()))

            #     policy_loss.append(torch.tensor(local_policy_loss).sum(-1))

            # policy_loss = torch.stack(policy_loss).cuda().requires_grad_(True)

            # policy_loss.register_hook(lambda grad: grad / config.UNROLL_STEPS)

            # accs['policy'].append(policy_loss)

            accs['value_diff'].append(torch.abs(torch.squeeze(value) - target_value[:, tstep]))
            # accs['reward_diff'].append(torch.abs(torch.squeeze(reward) - target_reward[:, tstep]))

            accs['value'].append(torch.squeeze(value))
            accs['policy'].append(torch.squeeze(policy_logits_flat))  # Use flattened policy for consistency
            # accs['reward'].append(torch.squeeze(reward))

            accs['target_value'].append(target_value[:, tstep])
            accs['target_policy'].append(target_policy[:, tstep])
            # accs['target_reward'].append(target_reward[:, tstep])

        if len(combats) > 0:
            obs, results = combats
            # Make sure combat observations have the right shape for the model
            # obs shape should be (batch_size, observation_size) = (batch_size, 5152)
            if obs.ndim == 4:  # If obs is (batch, 58, 4, 7), reshape to flat
                obs_flat = obs.reshape(obs.shape[0], -1)  # (batch, 58*4*7) = (batch, 1624)
                # Pad to match model's expected observation size (5152)
                if obs_flat.shape[1] < 5152:
                    padding = np.zeros((obs_flat.shape[0], 5152 - obs_flat.shape[1]))
                    obs_flat = np.concatenate([obs_flat, padding], axis=1)
                elif obs_flat.shape[1] > 5152:
                    obs_flat = obs_flat[:, :5152]  # Truncate if too large
            else:
                obs_flat = obs
            
            _, _, board_distribution = agent.initial_inference(obs_flat)
            torch_obs = torch.from_numpy(obs[:,0:58,:,:]).float().to(device)
            torch_results = torch.from_numpy(results).float().to(device)
            # from shape [batch] to shape [batch, 1 ,1 ,1]
            torch_results = torch.reshape(torch_results, (torch_results.shape[0], 1, 1, 1))
            
            # Compute board loss for combat data
            combat_board_loss = torch.sum(MAE_loss(board_distribution, torch_obs) * torch_results, dim=[1,2,3])

        accs = {k: torch.stack(v, -1) for k, v in accs.items()}

        mean_loss = value_loss.mean() + policy_loss.mean()
        if len(combats) > 0:
            mean_loss += combat_board_loss.mean()
        mean_loss.register_hook(lambda grad: grad * (1 / config.UNROLL_STEPS))

        # Leaving this here in case I want to use it later.
        # This was used in Atari but not in board games. Also, very unclear how to
        # Create the importance_weights from paper or from the source code.
        # loss = loss * importance_weights  # importance sampling correction
        # mean_loss = tf.math.divide_no_nan(
        #     tf.reduce_sum(loss), tf.reduce_sum(importance_weights))

        # if config.WEIGHT_DECAY > 0.:
        #     l2_loss = config.WEIGHT_DECAY * sum(
        #         self.l2_loss(p)
        #         for p in agent.parameters())
        # else:
        #     l2_loss = mean_loss * 0.

        # mean_loss += l2_loss

        sum_accs = {k: torch.sum(a, -1) for k, a in accs.items()}

        def get_mean(k):
            return torch.mean(sum_accs[k])

        summary_writer.add_scalar('prediction/value', get_mean('value'), train_step)
        # summary_writer.add_scalar('prediction/reward', get_mean('reward'), train_step)
        summary_writer.add_scalar('prediction/value_variance', torch.mean(torch.var(accs['value'], dim=0)), train_step)
        summary_writer.add_scalar('prediction/policy_variance', torch.mean(torch.var(accs['policy'], dim=1)), train_step)

        summary_writer.add_scalar('target/value', get_mean('target_value'), train_step)
        # summary_writer.add_scalar('target/reward', get_mean('target_reward'), train_step)
        summary_writer.add_scalar('target/value_variance', torch.mean(torch.var(accs['target_value'], dim=0)), train_step)
        summary_writer.add_scalar('target/policy_variance', torch.mean(torch.var(accs['target_policy'], dim=1)), train_step)

        summary_writer.add_scalar('losses/value', torch.mean(value_loss), train_step)
        if len(combats) > 0:
            summary_writer.add_scalar('losses/board', combat_board_loss.mean(), train_step)
        # summary_writer.add_scalar('losses/reward', get_mean('reward_loss'), train_step)
        summary_writer.add_scalar('losses/policy', torch.mean(policy_loss), train_step)
        # summary_writer.add_scalar('losses/directive', torch.mean(torch.sum(directive_loss, dim=0)), train_step)
        summary_writer.add_scalar('losses/total', torch.mean(mean_loss), train_step)
        # summary_writer.add_scalar('losses/l2', l2_loss, train_step)

        summary_writer.add_scalar('accuracy/value', -get_mean('value_diff'), train_step)
        # summary_writer.add_scalar('accuracy/reward', -get_mean('reward_diff'), train_step)

        # summary_writer.add_scalar('episode_max/reward', torch.max(target_reward), train_step)
        # summary_writer.add_scalar('episode_max/value', torch.max(target_value), train_step)
        summary_writer.flush()

        return mean_loss

    def scale_gradient(self, grad, scale):
        return scale * grad

    def l2_loss(self, t):
        return torch.sum(t ** 2) / 2
