import os
import heapq
import random
from dataclasses import dataclass
from typing import List, Optional, Type, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


def get_non_dominated_inds(points: np.ndarray) -> np.ndarray:
    """Return indices of non-dominated points (maximization assumed)."""
    is_dominated = np.zeros(len(points), dtype=bool)
    for i, p in enumerate(points):
        for j, q in enumerate(points):
            if i != j and np.all(q >= p) and np.any(q > p):
                is_dominated[i] = True
                break
    return np.where(~is_dominated)[0]


def hypervolume(ref_point: np.ndarray, front: np.ndarray) -> float:
    """Compute hypervolume for 2D front w.r.t. ref_point (assumes maximization on each objective)."""
    pts = front.copy()
    idx = np.argsort(-pts[:, 0])
    sorted_pts = pts[idx]
    hv = 0.0
    prev = ref_point[0]
    for a, b in sorted_pts:
        width = prev - a
        height = ref_point[1] - b
        hv += max(width, 0) * max(height, 0)
        prev = a
    return hv


def crowding_distance(points: np.ndarray) -> np.ndarray:
    """Compute the crowding distance of a set of points."""
    pts = (points - points.min(axis=0)) / (points.ptp(axis=0) + 1e-8)
    dim_sorted = np.argsort(pts, axis=0)
    sorted_pts = np.take_along_axis(pts, dim_sorted, axis=0)
    distances = np.abs(sorted_pts[:-2] - sorted_pts[2:])
    distances = np.pad(distances, ((1,), (0,)), constant_values=1)
    crowd = np.zeros(points.shape)
    crowd[dim_sorted, np.arange(points.shape[-1])] = distances
    return np.sum(crowd, axis=-1)


@dataclass
class Transition:
    observation: np.ndarray
    action: Union[float, int]
    reward: np.ndarray
    next_observation: np.ndarray
    terminal: bool


class BasePCNModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        reward_dim: int,
        scaling_factor: np.ndarray,
        hidden_dim: int,
        horizon_scaling: float = 1.0
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        sf = np.concatenate([scaling_factor, np.array([horizon_scaling], dtype=np.float32)])
        self.scaling_factor = nn.Parameter(torch.tensor(sf).float(), requires_grad=False)
        self.hidden_dim = hidden_dim

    def forward(self, state, desired_return, desired_horizon):
        c = torch.cat((desired_return, desired_horizon), dim=-1)
        c = c * self.scaling_factor
        s_emb = self.s_emb(state.float())
        c_emb = self.c_emb(c)
        return self.fc(s_emb * c_emb)


class DiscreteActionsDefaultModel(BasePCNModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.s_emb = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim), nn.Sigmoid())
        self.c_emb = nn.Sequential(nn.Linear(self.reward_dim + 1, self.hidden_dim), nn.Sigmoid())
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.LogSoftmax(dim=1)
        )


class ContinuousActionsDefaultModel(BasePCNModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.s_emb = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim), nn.Sigmoid())
        self.c_emb = nn.Sequential(nn.Linear(self.reward_dim + 1, self.hidden_dim), nn.Sigmoid())
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim)
        )


class PCNAgent:
    def __init__(
        self,
        env: gym.Env,
        scaling_factor: np.ndarray,
        horizon_scaling: float = 1.0,
        learning_rate: float = 1e-3,
        gamma: float = 1.0,
        batch_size: int = 256,
        hidden_dim: int = 64,
        noise: float = 0.1,
        log_dir: str = 'runs/PCN',
        seed: Optional[int] = None,
        device: Union[torch.device, str] = 'cpu',
        model_class: Optional[Type[BasePCNModel]] = None
    ):
        self.env = env
        self.device = torch.device(device)
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
        obs_space = env.observation_space.shape[0]
        act_space = env.action_space.shape[0] if isinstance(env.action_space, gym.spaces.Box) else env.action_space.n
        self.reward_dim = scaling_factor.shape[0]
        self.horizon_scaling = horizon_scaling
        self.continuous = isinstance(env.action_space, gym.spaces.Box)
        self.gamma = gamma
        self.batch_size = batch_size
        self.noise = noise
        self.scaling_factor = scaling_factor
        self.hidden_dim = hidden_dim
        self.global_step = 0

        if model_class is None:
            model_class = ContinuousActionsDefaultModel if self.continuous else DiscreteActionsDefaultModel
        self.model = model_class(
            obs_space,
            act_space,
            self.reward_dim,
            self.scaling_factor,
            self.hidden_dim,
            horizon_scaling=self.horizon_scaling
        ).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.writer = SummaryWriter(log_dir)
        self.experience = []

    def _add_episode(self, transitions: List[Transition], max_size: int, step: int):
        for i in range(len(transitions)-2, -1, -1):
            transitions[i].reward += self.gamma * transitions[i+1].reward
        item = (0, step, transitions)
        if len(self.experience) < max_size:
            heapq.heappush(self.experience, item)
        else:
            heapq.heappushpop(self.experience, item)

    def _nlargest(self, n, threshold=0.2):
        returns = np.array([e[2][0].reward for e in self.experience])
        distances = crowding_distance(returns)
        sma = np.where(distances <= threshold)[0]
        nd_idx = get_non_dominated_inds(returns)
        nd = returns[nd_idx]
        exp_returns = np.tile(returns[:, None, :], (1, len(nd), 1))
        l2 = -np.min(np.linalg.norm(exp_returns - nd[None], axis=-1), axis=1)
        l2[sma] *= 2
        idx = np.argsort(l2)
        top = [self.experience[i] for i in idx[-n:]]
        for i, e in enumerate(self.experience):
            self.experience[i] = (l2[i], e[1], e[2])
        heapq.heapify(self.experience)
        return top

    def _choose_commands(self, num_eps: int):
        eps = self._nlargest(num_eps)
        returns = np.array([e[2][0].reward for e in eps])
        horizons = np.array([len(e[2]) for e in eps])
        nd = get_non_dominated_inds(returns)
        returns, horizons = returns[nd], horizons[nd]
        i = np.random.randint(len(returns))
        des_h = float(horizons[i] - 2)
        des_r = returns[i].copy()
        m, s = des_r.mean(), des_r.std()
        obj = np.random.randint(len(des_r))
        des_r[obj] += np.random.uniform(high=s)
        return des_r.astype(np.float32), np.array([des_h], dtype=np.float32)

    def update(self):
        batch = random.choices(self.experience, k=self.batch_size)
        obs_list, acts_list, dr_list, dh_list = zip(*[(t.observation, t.action, t.reward, len(ep)-i)
                                               for _, _, ep in batch for i, t in enumerate(ep)])
        obs = torch.tensor(np.array(obs_list), device=self.device).float()
        dr = torch.tensor(np.stack(dr_list), device=self.device).float()
        dh = torch.tensor(np.array(dh_list, dtype=np.float32), device=self.device).unsqueeze(1)
        pred = self.model(obs, dr, dh)
        self.opt.zero_grad()
        if self.continuous:
            acts_tensor = torch.tensor(np.array(acts_list, dtype=np.float32), device=self.device)
            loss = F.mse_loss(acts_tensor, pred)
        else:
            acts_tensor = torch.tensor(acts_list, device=self.device)
            onehot = F.one_hot(acts_tensor, pred.size(-1)).float()
            loss = -(onehot * pred).sum(-1).mean()
        loss.backward()
        self.opt.step()
        self.writer.add_scalar('train/loss', loss.item(), self.global_step)
        self.global_step += 1
        return loss.item()

    def train(self,
              total_timesteps: int,
              eval_env: gym.Env,
              ref_point: np.ndarray,
              max_buffer_size: int = 100,
              num_er_episodes: int = 20,
              num_step_episodes: int = 10,
              num_model_updates: int = 50):
        self.experience.clear()
        step = 0
        for _ in range(num_er_episodes):
            obs, _ = self.env.reset()
            done = False
            traj = []
            while not done:
                action = self.env.action_space.sample()
                nobs, rew, term, trunc, _ = self.env.step(action)
                traj.append(Transition(obs, action, np.array(rew, np.float32), nobs, term))
                obs = nobs
                done = term or trunc
                step += 1
            self._add_episode(traj, max_buffer_size, step)

        while step < total_timesteps:
            for _ in range(num_model_updates):
                self.update()
            des_r, des_h = self._choose_commands(num_er_episodes)
            leaves = np.array([e[2][0].reward for e in self.experience[len(self.experience)//2:]])
            hv = hypervolume(ref_point, leaves)
            self.writer.add_scalar('train/hypervolume', hv, step)
            for _ in range(num_step_episodes):
                obs, _ = self.env.reset()
                done = False
                traj = []
                while not done:
                    action = self.model(
                        torch.tensor([obs], device=self.device).float(),
                        torch.tensor([des_r], device=self.device),
                        torch.tensor([des_h], device=self.device)
                    )
                    a = action.detach().cpu().numpy()[0] if self.continuous else action.exp().argmax(dim=1).item()
                    nobs, rew, term, trunc, _ = self.env.step(a)
                    traj.append(Transition(obs, a, np.array(rew, np.float32), nobs, term))
                    obs = nobs
                    done = term or trunc
                    step += 1
                self._add_episode(traj, max_buffer_size, step)

    def evaluate(self, num_episodes: int, eval_env: gym.Env = None, render: bool = False) -> np.ndarray:
        env = eval_env or self.env
        all_returns = []
        for ep in range(num_episodes):
            obs, _ = env.reset()
            done = False
            cum_return = np.zeros(self.reward_dim, dtype=np.float32)
            while not done:
                dr, dh = self._choose_commands(1)
                state = torch.tensor([obs], device=self.device).float()
                action_tensor = self.model(state, torch.tensor([dr], device=self.device), torch.tensor([dh], device=self.device))
                if self.continuous:
                    action = action_tensor.detach().cpu().numpy()[0]
                else:
                    action = action_tensor.exp().argmax(dim=1).item()
                obs, rew, term, trunc, _ = env.step(action)
                cum_return += np.array(rew, dtype=np.float32)
                done = term or trunc
                if render:
                    env.render()
            all_returns.append(cum_return)
            self.writer.add_scalar('eval/episode_return', float(cum_return.mean()), self.global_step)
        return np.mean(all_returns, axis=0)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'global_step': self.global_step
        }, filepath)

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
