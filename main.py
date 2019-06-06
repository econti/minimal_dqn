#!/usr/bin/env python3

"""
Usage:

$ . ~/env/bin/activate

Example pong command (~900k ts solve):
    python main.py \
        --env "PongNoFrameskip-v4" --CnnDQN --learning_rate 0.00001 \
        --target_update_rate 0.1 --replay_size 100000 --start_train_ts 10000 \
        --epsilon_start 1.0 --epsilon_end 0.01 --epsilon_decay 30000 --max_ts 1400000 \
        --batch_size 32 --gamma 0.99 --log_every 10000

Example cartpole command (~8k ts to solve):
    python main.py \
        --env "CartPole-v0" --learning_rate 0.001 --target_update_rate 0.1 \
        --replay_size 5000 --start_train_ts 32 --epsilon_start 1.0 --epsilon_end 0.01 \
        --epsilon_decay 500 --max_ts 10000 --batch_size 32 --gamma 0.99 --log_every 200
"""

import argparse
import math
import random
from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
from helpers import ReplayBuffer, make_atari, make_gym_env, wrap_deepmind, wrap_pytorch
from models import DQN, CnnDQN
from torch.distributions import normal


USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    print("Using GPU: GPU requested and available.")
    dtype = torch.cuda.FloatTensor
    dtypelong = torch.cuda.LongTensor
else:
    print("NOT Using GPU: GPU not requested or not available.")
    dtype = torch.FloatTensor
    dtypelong = torch.LongTensor


class Agent:
    def __init__(self, env, q_network, target_q_network):
        self.env = env
        self.q_network = q_network
        self.target_q_network = target_q_network
        self.num_actions = env.action_space.n

    def act(self, state, epsilon):
        """DQN action - max q-value w/ epsilon greedy exploration."""
        if random.random() > epsilon:
            state = torch.tensor(np.float32(state)).type(dtype).unsqueeze(0)
            q_value = self.q_network.forward(state)
            return q_value.max(1)[1].data[0]
        return torch.tensor(random.randrange(self.env.action_space.n))

    def bdqn_act(self, state, blr_params):
        """Bayesian DQN action - take dot product between linear regression
        weights and feature representation. Exploration done by sampling weights
        in training loop."""
        with torch.no_grad():
            state = torch.tensor(np.float32(state)).type(dtype).unsqueeze(0)
            features = self.q_network.forward(state)
            q_values = torch.mm(blr_params.E_W_, features.transpose(0, 1)).squeeze()
        return torch.argmax(q_values).data


def compute_td_loss(agent, batch_size, replay_buffer, optimizer, gamma):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    state = torch.tensor(np.float32(state)).type(dtype)
    next_state = torch.tensor(np.float32(next_state)).type(dtype)
    action = torch.tensor(action).type(dtypelong)
    reward = torch.tensor(reward).type(dtype)
    done = torch.tensor(done).type(dtype)

    q_values = agent.q_network(state)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    # double q-learning
    online_next_q_values = agent.q_network(next_state)
    _, max_indicies = torch.max(online_next_q_values, dim=1)
    target_q_values = agent.target_q_network(next_state)
    next_q_value = torch.gather(target_q_values, 1, max_indicies.unsqueeze(1))

    expected_q_value = reward + gamma * next_q_value.squeeze() * (1 - done)
    loss = (q_value - expected_q_value.data).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def get_epsilon(epsilon_start, epsilon_final, epsilon_decay, frame_idx):
    return epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1.0 * frame_idx / epsilon_decay
    )


def soft_update(q_network, target_q_network, tau):
    for t_param, param in zip(target_q_network.parameters(), q_network.parameters()):
        if t_param is param:
            continue
        new_param = tau * param.data + (1.0 - tau) * t_param.data
        t_param.data.copy_(new_param)


def hard_update(q_network, target_q_network):
    for t_param, param in zip(target_q_network.parameters(), q_network.parameters()):
        if t_param is param:
            continue
        new_param = param.data
        t_param.data.copy_(new_param)


class BLR:
    """
    Initialize BLR matrices.
    """

    def __init__(self, num_actions, last_layer_units):
        self.last_layer_units = last_layer_units
        # forgetting factor 1 = forget
        self.alpha = 1.0
        # prior for weight variance
        self.sigma = 0.001
        # prior for noise variance
        self.sigma_n = 1.0
        self.eye = torch.eye(last_layer_units).type(dtype)
        # Prior distribution on BLR weights
        dist = normal.Normal(loc=0, scale=0.01)
        # BLR weights from prior
        self.E_W = dist.sample((num_actions, last_layer_units)).type(dtype)
        # target BLR weights from prior
        self.E_W_target = dist.sample((num_actions, last_layer_units)).type(dtype)
        # BLR weights to be used when doing Thompson Sampling for exploration ??
        self.E_W_ = dist.sample((num_actions, last_layer_units)).type(dtype)
        # Each action has a covariance matrix for its weights in the BLR
        self.Cov_W = torch.eye(last_layer_units).repeat(num_actions, 1, 1).type(dtype)
        self.Cov_W_decom = self.Cov_W
        self.Cov_W_target = self.Cov_W
        self.phiphiT = torch.zeros(
            (num_actions, last_layer_units, last_layer_units)
        ).type(dtype)
        self.phiY = torch.zeros((num_actions, last_layer_units)).type(dtype)


def update_bayes_reg_posterior(blr_params, params, replay_buffer, agent):
    with torch.no_grad():
        # Forgetting parameter alpha suggest how much of the moment from the past
        # can be used, we set alpha to 1 which means do not use the past moment
        blr_params.phiphiT *= 1 - blr_params.alpha
        blr_params.phiY *= 1 - blr_params.alpha
        for _ in range(params.target_batch_size):
            # sample a minibatch of size one from replay buffer
            state, action, reward, next_state, done = replay_buffer.sample(1)
            state = torch.tensor(np.float32(state[0])).type(dtype).unsqueeze(0)
            next_state = (
                torch.tensor(np.float32(next_state[0])).type(dtype).unsqueeze(0)
            )
            action = action[0]
            reward = torch.tensor(reward[0]).type(dtype)
            done = torch.tensor(done[0]).type(dtype)

            blr_params.phiphiT[int(action)] += torch.mm(
                agent.q_network(state).transpose(0, 1), agent.q_network(state)
            )
            target_q_values = torch.mm(
                blr_params.E_W_target,
                agent.target_q_network(next_state).transpose(0, 1),
            )
            max_target_q = torch.max(target_q_values)
            blr_params.phiY[int(action)] += agent.q_network(state).transpose(
                0, 1
            ).squeeze() * (reward + (1.0 - done) * params.gamma * max_target_q)

        for i in range(agent.num_actions):
            inv = (
                blr_params.phiphiT[i] / blr_params.sigma_n
                + 1 / blr_params.sigma * blr_params.eye
            ).inverse()
            blr_params.E_W[i] = (
                torch.mm(inv, blr_params.phiY[i].unsqueeze(1)).squeeze()
                / blr_params.sigma_n
            )
            blr_params.Cov_W[i] = blr_params.sigma * inv


def sample_W(num_actions, blr_params):
    """
    Thompson sampling - sample model W form the posterior for exploration.
    """
    dist = normal.Normal(loc=0, scale=1)
    for i in range(num_actions):
        sample = dist.sample((blr_params.last_layer_units, 1)).type(dtype)
        blr_params.E_W_[i] = (
            blr_params.E_W[i] + torch.mm(blr_params.Cov_W_decom[i], sample)[:, 0]
        )


def run_gym(params):
    if params.CnnDQN:
        env = make_atari(params.env)
        env = wrap_pytorch(wrap_deepmind(env))
        q_network = CnnDQN(
            env.observation_space.shape, env.action_space.n, params.BayesianDQN
        )
        target_q_network = deepcopy(q_network)
    else:
        env = make_gym_env(params.env)
        q_network = DQN(
            env.observation_space.shape, env.action_space.n, params.BayesianDQN
        )
        target_q_network = deepcopy(q_network)

    if params.BayesianDQN:
        blr = BLR(env.action_space.n, q_network.fc[0].out_features)

    if USE_CUDA:
        q_network = q_network.cuda()
        target_q_network = target_q_network.cuda()

    agent = Agent(env, q_network, target_q_network)
    optimizer = optim.Adam(q_network.parameters(), lr=params.learning_rate)
    replay_buffer = ReplayBuffer(params.replay_size)

    losses, all_rewards = [], []
    episode_reward = 0
    state = env.reset()

    for ts in range(1, params.max_ts + 1):
        epsilon = get_epsilon(
            params.epsilon_start, params.epsilon_end, params.epsilon_decay, ts
        )

        if params.BayesianDQN:
            action = agent.bdqn_act(state, blr)
        else:
            action = agent.act(state, epsilon)

        next_state, reward, done, _ = env.step(int(action.cpu()))
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        # Thompson sampling BLR weights for exploration
        if params.BayesianDQN and ts % params.f_sampling == 0:
            sample_W(agent.num_actions, blr)

        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0

        if len(replay_buffer) > params.start_train_ts:
            # Update the q-network & the target network
            loss = compute_td_loss(
                agent, params.batch_size, replay_buffer, optimizer, params.gamma
            )
            losses.append(loss.data)
            soft_update(
                agent.q_network, agent.target_q_network, params.target_update_rate
            )

            target_updates_since_posterior_update = 0
            if ts % params.target_network_update_f == 0:
                target_updates_since_posterior_update += 1
                if (
                    params.BayesianDQN
                    and target_updates_since_posterior_update
                    == params.target_W_update_f
                ):
                    # update the posterior distribution of W
                    update_bayes_reg_posterior(blr, params, replay_buffer, agent)
                    blr.E_W_target = blr.E_W
                    blr.Cov_W_target = blr.Cov_W
                    target_updates_since_posterior_update = 0
                    for i in range(agent.num_actions):
                        blr.Cov_W_decom[i] = torch.tensor(
                            np.linalg.cholesky(
                                ((blr.Cov_W[i] + blr.Cov_W[i].transpose(0, 1))) / 2.0
                            )
                            .cpu()
                            .numpy()
                        ).type(dtype)

        if ts % params.log_every == 0:
            out_str = "Timestep {}".format(ts)
            if len(all_rewards) > 0:
                out_str += ", Reward: {}".format(all_rewards[-1])
            if len(losses) > 0:
                out_str += ", TD Loss: {}".format(losses[-1])
            print(out_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--CnnDQN", action="store_true")
    parser.add_argument("--BayesianDQN", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--target_update_rate", type=float, default=0.1)
    parser.add_argument("--replay_size", type=int, default=100000)
    parser.add_argument("--start_train_ts", type=int, default=10000)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--epsilon_decay", type=int, default=30000)
    parser.add_argument("--max_ts", type=int, default=1400000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--log_every", type=int, default=10000)
    # Bayesian DQN arguments
    # -----------------------
    # how often to sample BLR weights for exploration via. Thompson Sampling (E_W_ weights)
    parser.add_argument("--f_sampling", type=int, default=1000)
    # how often to update BLR weights (E_W) and covariance (Cov)
    parser.add_argument("--target_network_update_f", type=int, default=10000)
    parser.add_argument("--target_W_update_f", type=int, default=10)
    # ??????????
    parser.add_argument("--target_batch_size", type=int, default=5000)
    run_gym(parser.parse_args())
