from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from learning.utils.env import launch_env
from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper
from learning.utils.teacher import PurePursuitExpert

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ExpertTrajDataset(Dataset):
    
    def __init__(self,args, transform=None):
        
        self.file_dir = args.data_directory
#         self.transform = transform
        self.episodes = args.episodes

        observation = torch.FloatTensor(torch.load('{}/data_o_{}.pt'.format(self.file_dir, 0)))
        self.observation_dim = np.prod(observation.shape[1:])
        action = torch.FloatTensor(torch.load('{}/data_a_{}.pt'.format(self.file_dir, 0)))
        self.action_dim = np.prod(action.shape[1:])
        
    def __len__(self):
        return self.episodes
    
    def __getitem__(self,idx):
        if torch .is_tensor(idx):
            idx = idx.tolist()
            
        observation = torch.load('{}/data_o_{}.pt'.format(self.file_dir, idx))   
        action = torch.load('{}/data_a_{}.pt'.format(self.file_dir,idx))   

        sample = {'observation': observation, 'action':action}
        
        return sample
    
def generate_expert_trajectorys(args):

    env = launch_env()
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env) 
    env = ImgWrapper(env)
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    print("Initialized Wrappers")

    expert = PurePursuitExpert(env=env)

    for episode in range(0, args.episodes):
        print("Starting episode", episode)
        observations = []
        actions = []
        for steps in range(0, args.steps):
            # use our 'expert' to predict the next action.
            action = expert.predict(None)
            observation, reward, done, info = env.step(action)
            observations.append(observation)
            actions.append(action)
            # env.render()
        env.reset()
        torch.save(actions, '{}/data_a_{}.pt'.format(args.data_directory,episode))
        torch.save(observations, '{}/data_o_{}.pt'.format(args.data_directory,episode))    
    
    env.close()

def generate_policy_trajectories(policy, episodes, steps):
    env = launch_env()
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env) 
    env = ImgWrapper(env)
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    
    print("Initialized Wrappers")
    observations = []
    actions = []
    masks = []
    for episode in range(0,episodes):
        observation = env.reset()
        for step in range(0, steps):
            observation = torch.from_numpy(observation).float().to(device).unsqueeze(0)
            action = policy(observation)
            action = action.squeeze().data.cpu().numpy()

            observation, reward, done, info = env.step(action)
            observations.append(observation)
            actions.append(action)

            mask = 0 if done or step == steps-1 else 1
            masks.append(mask)
            # env.render()
    env.close()

    return observations, actions, masks

