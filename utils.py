from collections import defaultdict
import os
import random
import time
import tqdm

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch.utils.data as td


class EvalDataset(td.Dataset):
    def __init__(self, positive_data, item_num, positive_mat, negative_samples=99):
        super(EvalDataset, self).__init__()
        self.positive_data = np.array(positive_data)
        self.item_num = item_num
        self.positive_mat = positive_mat
        self.negative_samples = negative_samples
        
        self.reset()
        
    def reset(self):
        print("Resetting dataset")
        data = self.create_valid_data()
        labels = np.zeros(len(self.positive_data) * (1 + self.negative_samples))
        labels[::1+self.negative_samples] = 1
        self.data = np.concatenate([
            np.array(data), 
            np.array(labels)[:, np.newaxis]], 
            axis=1
        )

    def create_valid_data(self):
        valid_data = []
        for user, positive in self.positive_data:
            valid_data.append([user, positive])
            for i in range(self.negative_samples):
                negative = np.random.randint(self.item_num)
                while (user, negative) in self.positive_mat:
                    negative = np.random.randint(self.item_num)
                    
                valid_data.append([user, negative])
        return valid_data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user, item, label = self.data[idx]
        output = {
            "user": user,
            "item": item,
            "label": np.float32(label),
        }
        return output


#https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_dim, mu=0.0, theta=0.15, max_sigma=0.4, min_sigma=0.4, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_dim
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return torch.tensor([action + ou_state]).float()


class Prioritized_Buffer(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity   = capacity
        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def push(self, user, memory, action, reward, next_user, next_memory, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((user, memory, action, reward, next_user, next_memory, done))
        else:
            self.buffer[self.pos] = (user, memory, action, reward, next_user, next_memory, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs  = prios ** self.prob_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total    = len(self.buffer)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)

        batch       = list(zip(*samples))
        user        = np.concatenate(batch[0])
        memory      = np.concatenate(batch[1])
        action      = batch[2]
        reward      = batch[3]
        next_user   = np.concatenate(batch[4])
        next_memory = np.concatenate(batch[5])
        done        = batch[6]

        return user, memory, action, reward, next_user, next_memory, done

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


def get_beta(idx, beta_start=0.4, beta_steps=100000):
    return min(1.0, beta_start + idx * (1.0 - beta_start) / beta_steps)

def preprocess_data(data_dir, train_rating):
    data = pd.read_csv(os.path.join(data_dir, train_rating), 
                       sep='\t', header=None, names=['user', 'item', 'rating'], 
                       usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.int8})
    data = data[data['rating'] > 3][['user', 'item']]
    user_num = data['user'].max() + 1
    item_num = data['item'].max() + 1

    train_data = data.sample(frac=0.8, random_state=16)
    test_data = data.drop(train_data.index).values.tolist()
    train_data = train_data.values.tolist()

    train_mat = defaultdict(int)
    test_mat = defaultdict(int)
    for user, item in train_data:
        train_mat[user, item] = 1.0
    for user, item in test_data:
        test_mat[user, item] = 1.0
    train_matrix = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    dict.update(train_matrix, train_mat)
    test_matrix = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    dict.update(test_matrix, test_mat)
    
    appropriate_users = np.arange(user_num).reshape(-1, 1)[(train_matrix.sum(1) >= 20)]
    
    return (train_data, train_matrix, test_data, test_matrix, 
            user_num, item_num, appropriate_users)

def to_np(tensor):
    return tensor.detach().cpu().numpy()

def hit_metric(recommended, actual):
    return int(actual in recommended)

def dcg_metric(recommended, actual):
    if actual in recommended:
        index = recommended.index(actual)
        return np.reciprocal(np.log2(index + 2))
    return 0
