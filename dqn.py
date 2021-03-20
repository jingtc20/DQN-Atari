#!/usr/bin/env python3
from lib import wrappers
from lib import dqn_model
import parameters
from  parameters import params

import argparse
import time
import numpy as np
import collections
import os
import errno
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F


is_cuda = True
NAME = parameters.NAME  
DEFAULT_ENV_NAME = "{}NoFrameskip-v4".format(NAME)  

UPDATE_FRE = params['UPDATE_FRE']
STACK_FRAME = params['STACK_FRAME']
MAX_EP =  params['MAX_EP']
SAVE_EP = params['SAVE_EP']
IS_STEPLR, is_huber = params['IS_STEPLR'], params['is_huber']
LR_GAMMA, STEP_LR_MIN, STEP_EP_REWARD = params['LR_GAMMA'], params['STEP_LR_MIN'], params['STEP_EP_REWARD']
IS_CLIP_REWARD, IS_CLIP_LOSS = params['is_clip_reward'], params['is_clip_loss']

LEARNING_RATE = 1e-4
GAMMA = 0.99  
BATCH_SIZE = 32
REPLAY_SIZE =  10000  
REPLAY_START_SIZE = 10000
SYNC_TARGET_FRAMES = 1000 * UPDATE_FRE
EPSILON_DECAY_LAST_FRAME = 150000 * UPDATE_FRE
EPSILON_START = 1.0
EPSILON_FINAL = 0.01
SKIP_FRAME = 4
out_dir = 'models'
data_dir = 'data'

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


def make_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones), np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()
        self.is_terminal = False
        self.last_lives = 0

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    def clip_reward(self, reward):
        if reward > 0:
            return 1
        elif reward < 0:
            return -1
        else:
            return 0

    def play_step(self, net, IS_CLIP, epsilon=0.0, device="cpu"):
        done_reward = None
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        new_state, reward, is_done, info = self.env.step(action)
        self.total_reward += reward
        if info['ale.lives'] < self.last_lives:
            self.is_terminal = True
        else:
            self.is_terminal = is_done

        if IS_CLIP:  # clip reward
            reward = self.clip_reward(reward)
        exp = Experience(self.state, action, reward, self.is_terminal, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        self.last_lives = info['ale.lives']

        if is_done:
            done_reward = self.total_reward
            self._reset()         
        return done_reward  # return None if not done else total_reward in the episode.


def calc_loss(batch, net, tgt_net, gamma, is_huber, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.tensor(dones).to(device)    

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * gamma + rewards_v
    if is_huber:
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    else:
        loss = nn.MSELoss()(state_action_values, expected_state_action_values)
    return loss



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=is_cuda, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME, help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    make_dir(out_dir)  # save models
    make_dir(data_dir)  # save datas
    print('device:', device,  'BETA: ', BETA, 'NAME: ', NAME)
    print('clip_reward:',IS_CLIP_REWARD,'clip_loss:', IS_CLIP_LOSS,'is_huber:', is_huber)

    env = wrappers.make_env(args.env, SKIP_FRAME, STACK_FRAME)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    
    epsilon = EPSILON_START
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE) 
    scheduler = StepLR(optimizer, step_size=10000, gamma=LR_GAMMA)   
    optimizer_tgt = optim.Adam(tgt_net.parameters(), lr=LEARNING_RATE)   
    total_rewards, mean_return, train_times, total_losses  = [], [], [], []
    frame_idx, ts_frame, episode, ep_time, ep_loss, best_mean_reward = 0, 0, 0, 0, 0, -21
    ts = time.time()


    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
        reward = agent.play_step(net, IS_CLIP_REWARD, epsilon, device=device)

        if reward is not None:    # the episode is done
            ep_time = time.time() - ts
            episode += 1
            total_rewards.append(reward)
            total_losses.append(ep_loss)
            speed = (frame_idx - ts_frame) / ep_time
            ts_frame = frame_idx
            ts = time.time()
            ep_loss = 0
            mean_reward = np.mean(total_rewards[-100:])
            mean_return.append(mean_reward) 
            print("%d: done %d games, mean reward %.2f, reward %.2f, lr %.5f, loss %.2f, time %.2f s, eps %.2f, speed %.2f f/s" % (
                frame_idx, episode, mean_reward, reward, scheduler.get_last_lr()[0], total_losses[-1], ep_time, epsilon, speed))            
            if best_mean_reward < mean_reward:
                if episode > SAVE_EP:
                    torch.save(net.state_dict(), '{}/episode_{}'.format(out_dir, episode))
                print("mean reward %.2f -> %.2f" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if episode == MAX_EP:
                print("Finished in %d frames!" % frame_idx)
                break


        if len(buffer) >= REPLAY_START_SIZE and frame_idx % UPDATE_FRE == 0:
            optimizer.zero_grad()
            batch = buffer.sample(BATCH_SIZE)
            loss_t = calc_loss(batch, net, tgt_net, GAMMA, is_huber, device=device)
            ep_loss += loss_t.item()
            loss_t.backward()
            if IS_CLIP_LOSS:
                for parameter in net.parameters():    
                    parameter.grad.data.clamp_(-1,1)
            optimizer.step()  
            if frame_idx % SYNC_TARGET_FRAMES == 0:
                tgt_net.load_state_dict(net.state_dict())
            if IS_STEPLR and scheduler.get_last_lr()[0] > STEP_LR_MIN and mean_reward >= STEP_EP_REWARD:
                scheduler.step()     

        if episode % data_save_fre == 0 and episode >= save_ep_min:
            np.save('{}/mean_return_np_{}'.format(data_dir, episode), np.array(mean_return))
            np.save('{}/total_rewards_np_{}'.format(data_dir, episode), np.array(total_rewards))
            np.save('{}/loss_{}'.format(data_dir, episode), np.array(total_losses)) 
            
    plt.figure(1)
    plt.title('episode return in {}'.format(NAME))
    plt.ylabel('epoch_return', fontsize=14)
    plt.xlabel('epoch', fontsize=14)
    plt.plot(list(range(MAX_EP)), total_rewards, label="DQN")
    plt.legend( loc='best')
    plt.savefig('episode_return_dqn_{}.png'.format(NAME))

    plt.figure(2)
    plt.title('mean return in {}'.format(NAME))
    plt.ylabel('mean_return', fontsize=14)
    plt.xlabel('epoch', fontsize=14)
    plt.plot(list(range(MAX_EP)), mean_return, label="DQN")
    plt.legend( loc='best')
    plt.savefig('mean_return_dqn_{}.png'.format(NAME))

    plt.figure(3)
    plt.title('episode loss in {}'.format(NAME))
    plt.ylabel('epoch_loss', fontsize=14)
    plt.xlabel('epoch', fontsize=14)
    plt.plot(list(range(MAX_EP)), total_losses, label="DQN")
    plt.legend( loc='best')
    plt.savefig('episode_loss_dqn_{}.png'.format(NAME))





    
