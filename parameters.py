 
import matplotlib.pyplot as plt
import numpy as np
import random
import gym
import math
import torch

NAME = "Breakout"
# NAME = "MsPacman"
# NAME = 'Pong'

if NAME == "Breakout":
    params = {
        'UPDATE_FRE': 4,
        'STACK_FRAME': 2,
        'MAX_EP':  4001,
        'SAVE_EP': 3000,
        'IS_STEPLR': False,
        'is_huber': False,
        'LR_GAMMA': 0.987,
        'STEP_LR_MIN': 1e-5,
        'STEP_EP_REWARD': 200,
        'data_save_fre': 1000, 
        'save_ep_min': 3000, 
        'is_clip_reward': True,
        'is_clip_loss': True,
    }
elif NAME == "MsPacman":
    params = {
        'UPDATE_FRE': 1,
        'STACK_FRAME': 4,
        'MAX_EP':  3001,
        'SAVE_EP': 2000,
        'IS_STEPLR': True,
        'is_huber': True,
        'LR_GAMMA': 0.987,
        'STEP_LR_MIN': 1e-5,
        'STEP_EP_REWARD': 0,
        'data_save_fre': 1000, 
        'save_ep_min': 1000,  
        'is_clip_reward': False,
        'is_clip_loss': False,
    }
elif NAME ==  "Pong": 
    params = {
        'UPDATE_FRE': 1,
        'STACK_FRAME': 4,
        'MAX_EP':  501,
        'SAVE_EP': 400,
        'IS_STEPLR': False,
        'is_huber': False,
        'LR_GAMMA': 0.98,
        'STEP_LR_MIN': 1e-4,
        'STEP_EP_REWARD': None,
        'data_save_fre': 100, 
        'save_ep_min': 400,  
        'is_clip_reward': True,
        'is_clip_loss': True,
    }


