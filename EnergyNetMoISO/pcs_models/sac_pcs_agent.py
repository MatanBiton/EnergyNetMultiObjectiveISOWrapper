"""
SAC-based PCS Agent for EnergyNet PCS Unit training.

This module provides a SAC-based PCS agent that inherits from GenericPCSAgent
and can be trained on PCSUnitEnv. It includes all the optimizations from the
multi-objective SAC implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, LinearLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import copy
import os
from typing import Tuple, Dict, Any, Optional, List, Union
from collections import deque
import random
import gymnasium as gym
import math

from .generic_pcs_agent import GenericPCSAgent


def orthogonal_init(module: nn.Module, gain: float = 1.0):
    """Apply orthogonal initialization to linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def xavier_uniform_init(module: nn.Module):
    """Apply Xavier uniform initialization to linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class RunningMeanStd:
    """Running mean and standard deviation tracker for reward normalization."""
    
    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-4):
        self.shape = shape
        self.epsilon = epsilon
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0
    
    def update(self, x: np.ndarray):
        """Update running statistics with new data."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int):
        """Update from pre-computed moments."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize data using current statistics."""
        return (x - self.mean) / np.sqrt(self.var + self.epsilon)
    
    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Denormalize data using current statistics."""
        return x * np.sqrt(self.var + self.epsilon) + self.mean


class ReplayBuffer:
    """Experience replay buffer for SAC."""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def push(self, state: np.ndarray, action: np.ndarray, reward: float, 
             next_state: np.ndarray, done: bool):
        """Add experience to buffer."""
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch from buffer."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays first for efficiency
        states_np = np.array(states)
        actions_np = np.array(actions)
        rewards_np = np.array(rewards)
        next_states_np = np.array(next_states)
        dones_np = np.array(dones)
        
        return (
            torch.from_numpy(states_np).float(),
            torch.from_numpy(actions_np).float(),
            torch.from_numpy(rewards_np).float().unsqueeze(1),
            torch.from_numpy(next_states_np).float(),
            torch.from_numpy(dones_np).bool().unsqueeze(1)
        )
    
    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """Actor network for continuous action spaces."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256],
                 log_std_min: float = -20, log_std_max: float = 2, 
                 use_orthogonal_init: bool = True, orthogonal_gain: float = 1.0):
        super(Actor, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Build network
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])
            input_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(input_dim, action_dim)
        self.log_std_head = nn.Linear(input_dim, action_dim)
        
        # Initialize weights
        if use_orthogonal_init:
            self.apply(lambda m: orthogonal_init(m, orthogonal_gain))
            # Use smaller gain for output layers
            orthogonal_init(self.mean_head, 0.01)
            orthogonal_init(self.log_std_head, 0.01)
        else:
            self.apply(xavier_uniform_init)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and log_std."""
        x = self.backbone(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforce action bounds
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
    """Critic network for single-objective rewards."""
    
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [256, 256], 
                 use_orthogonal_init: bool = True, orthogonal_gain: float = 1.0):
        super(Critic, self).__init__()
        
        # Build network
        layers = []
        input_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])
            input_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        self.output_head = nn.Linear(input_dim, 1)  # Single Q-value
        
        # Initialize weights
        if use_orthogonal_init:
            self.apply(lambda m: orthogonal_init(m, orthogonal_gain))
            # Use smaller gain for output layer
            orthogonal_init(self.output_head, 0.01)
        else:
            self.apply(xavier_uniform_init)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Q-value."""
        x = torch.cat([state, action], dim=1)
        x = self.backbone(x)
        return self.output_head(x)


class SACPCSAgent(GenericPCSAgent):
    """SAC-based PCS Agent with all optimizations."""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 action_bounds: Tuple[float, float] = (-1.0, 1.0),
                 # Network parameters
                 actor_hidden_dims: List[int] = [256, 256],
                 critic_hidden_dims: List[int] = [256, 256],
                 # Optimizer parameters
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 alpha_lr: float = 3e-4,
                 # Algorithm parameters
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 auto_tune_alpha: bool = True,
                 target_entropy: Optional[float] = None,
                 # Buffer parameters
                 buffer_capacity: int = 1000000,
                 batch_size: int = 256,
                 # Optimization features
                 use_lr_annealing: bool = False,
                 lr_annealing_type: str = 'cosine',
                 lr_annealing_steps: Optional[int] = None,
                 lr_min_factor: float = 0.1,
                 lr_decay_rate: float = 0.95,
                 use_reward_scaling: bool = False,
                 reward_scale_epsilon: float = 1e-4,
                 use_orthogonal_init: bool = False,
                 orthogonal_gain: float = 1.0,
                 actor_orthogonal_gain: float = 0.01,
                 critic_orthogonal_gain: float = 1.0,
                 use_value_clipping: bool = False,
                 value_clip_range: float = 200.0,
                 # Logging
                 device: str = 'auto',
                 verbose: bool = False,
                 tensorboard_log: Optional[str] = None):
        
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.verbose = verbose
        
        # Optimization features
        self.use_lr_annealing = use_lr_annealing
        self.lr_annealing_type = lr_annealing_type
        self.lr_annealing_steps = lr_annealing_steps
        self.lr_min_factor = lr_min_factor
        self.lr_decay_rate = lr_decay_rate
        self.use_reward_scaling = use_reward_scaling
        self.use_orthogonal_init = use_orthogonal_init
        self.orthogonal_gain = orthogonal_gain
        self.actor_orthogonal_gain = actor_orthogonal_gain
        self.critic_orthogonal_gain = critic_orthogonal_gain
        self.use_value_clipping = use_value_clipping
        self.value_clip_range = value_clip_range
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        if self.verbose:
            print(f"Using device: {self.device}")
            print(f"Optimization features enabled:")
            print(f"  LR Annealing: {self.use_lr_annealing}")
            print(f"  Reward Scaling: {self.use_reward_scaling}")
            print(f"  Orthogonal Init: {self.use_orthogonal_init}")
            print(f"  Value Clipping: {self.use_value_clipping}")
        
        # Reward scaling statistics
        if self.use_reward_scaling:
            self.reward_rms = RunningMeanStd(shape=(1,), epsilon=reward_scale_epsilon)
        
        # Networks
        self.actor = Actor(
            state_dim, action_dim, actor_hidden_dims,
            use_orthogonal_init=use_orthogonal_init,
            orthogonal_gain=actor_orthogonal_gain
        ).to(self.device)
        
        self.critic1 = Critic(
            state_dim, action_dim, critic_hidden_dims,
            use_orthogonal_init=use_orthogonal_init,
            orthogonal_gain=critic_orthogonal_gain
        ).to(self.device)
        
        self.critic2 = Critic(
            state_dim, action_dim, critic_hidden_dims,
            use_orthogonal_init=use_orthogonal_init,
            orthogonal_gain=critic_orthogonal_gain
        ).to(self.device)
        
        # Target networks
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        # Learning rate schedulers
        self.actor_scheduler = None
        self.critic1_scheduler = None
        self.critic2_scheduler = None
        self.alpha_scheduler = None
        
        if self.use_lr_annealing and self.lr_annealing_steps is not None:
            if self.lr_annealing_type == 'cosine':
                self.actor_scheduler = CosineAnnealingLR(
                    self.actor_optimizer, T_max=self.lr_annealing_steps,
                    eta_min=actor_lr * self.lr_min_factor
                )
                self.critic1_scheduler = CosineAnnealingLR(
                    self.critic1_optimizer, T_max=self.lr_annealing_steps,
                    eta_min=critic_lr * self.lr_min_factor
                )
                self.critic2_scheduler = CosineAnnealingLR(
                    self.critic2_optimizer, T_max=self.lr_annealing_steps,
                    eta_min=critic_lr * self.lr_min_factor
                )
            elif self.lr_annealing_type == 'linear':
                self.actor_scheduler = LinearLR(
                    self.actor_optimizer, start_factor=1.0,
                    end_factor=self.lr_min_factor, total_iters=self.lr_annealing_steps
                )
                self.critic1_scheduler = LinearLR(
                    self.critic1_optimizer, start_factor=1.0,
                    end_factor=self.lr_min_factor, total_iters=self.lr_annealing_steps
                )
                self.critic2_scheduler = LinearLR(
                    self.critic2_optimizer, start_factor=1.0,
                    end_factor=self.lr_min_factor, total_iters=self.lr_annealing_steps
                )
            elif self.lr_annealing_type == 'exponential':
                self.actor_scheduler = ExponentialLR(
                    self.actor_optimizer, gamma=self.lr_decay_rate
                )
                self.critic1_scheduler = ExponentialLR(
                    self.critic1_optimizer, gamma=self.lr_decay_rate
                )
                self.critic2_scheduler = ExponentialLR(
                    self.critic2_optimizer, gamma=self.lr_decay_rate
                )
        
        # Entropy regularization
        self.auto_tune_alpha = auto_tune_alpha
        if auto_tune_alpha:
            if target_entropy is None:
                self.target_entropy = -action_dim
            else:
                self.target_entropy = target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
            
            # Alpha scheduler
            if self.use_lr_annealing and self.lr_annealing_steps is not None:
                if self.lr_annealing_type == 'cosine':
                    self.alpha_scheduler = CosineAnnealingLR(
                        self.alpha_optimizer, T_max=self.lr_annealing_steps,
                        eta_min=alpha_lr * self.lr_min_factor
                    )
                elif self.lr_annealing_type == 'linear':
                    self.alpha_scheduler = LinearLR(
                        self.alpha_optimizer, start_factor=1.0,
                        end_factor=self.lr_min_factor, total_iters=self.lr_annealing_steps
                    )
                elif self.lr_annealing_type == 'exponential':
                    self.alpha_scheduler = ExponentialLR(
                        self.alpha_optimizer, gamma=self.lr_decay_rate
                    )
        else:
            self.alpha = alpha
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity, state_dim, action_dim)
        
        # Logging
        self.tensorboard_log = tensorboard_log
        if tensorboard_log:
            self.writer = SummaryWriter(log_dir=tensorboard_log)
        
        # Training statistics
        self.training_step = 0
        self.episode_count = 0
    
    def get_alpha(self) -> float:
        """Get current alpha value."""
        if self.auto_tune_alpha:
            return self.log_alpha.exp().item()
        else:
            return self.alpha
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action from policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if deterministic:
                mean, _ = self.actor(state_tensor)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(state_tensor)
            
            # Scale action to environment bounds
            action = action.cpu().numpy()[0]
            action = self.scale_action(action)
            
        return action
    
    def predict(self, obs, deterministic=True, **kwargs):
        """
        Predict the action for the given observation (GenericPCSAgent interface).
        
        Args:
            obs: The observation from the environment
            deterministic: Whether to use deterministic action selection
            **kwargs: Additional keyword arguments
            
        Returns:
            tuple: (action, state) where state is None for stateless agents
        """
        action = self.select_action(obs, deterministic)
        return torch.tensor(action, dtype=torch.float32), None
    
    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """Scale action from [-1, 1] to environment bounds."""
        low, high = self.action_bounds
        return low + (action + 1.0) * 0.5 * (high - low)
    
    def unscale_action(self, action: np.ndarray) -> np.ndarray:
        """Unscale action from environment bounds to [-1, 1]."""
        low, high = self.action_bounds
        return 2.0 * (action - low) / (high - low) - 1.0
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store transition in replay buffer."""
        # Unscale action for storage
        action_unscaled = self.unscale_action(action)
        
        # Apply reward scaling if enabled
        if self.use_reward_scaling:
            # Update reward statistics
            self.reward_rms.update(np.array([reward]).reshape(1, -1))
            # Store normalized reward
            scaled_reward = self.reward_rms.normalize(np.array([reward]))[0]
        else:
            scaled_reward = reward
        
        self.replay_buffer.push(state, action_unscaled, scaled_reward, next_state, done)
    
    def update(self) -> Dict[str, float]:
        """Update the networks."""
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.get_alpha() * next_log_probs
            target_q = rewards + (1 - dones.float()) * self.gamma * target_q
            
            # Apply value clipping if enabled
            if self.use_value_clipping:
                target_q = torch.clamp(target_q, -self.value_clip_range, self.value_clip_range)
        
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        actions_new, log_probs = self.actor.sample(states)
        q1_new = self.critic1(states, actions_new)
        q2_new = self.critic2(states, actions_new)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.get_alpha() * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha
        alpha_loss = torch.tensor(0.0)
        if self.auto_tune_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        # Update target networks
        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)
        
        # Update learning rate schedulers
        if self.use_lr_annealing:
            if self.actor_scheduler:
                self.actor_scheduler.step()
            if self.critic1_scheduler:
                self.critic1_scheduler.step()
            if self.critic2_scheduler:
                self.critic2_scheduler.step()
            if self.alpha_scheduler:
                self.alpha_scheduler.step()
        
        self.training_step += 1
        
        # Log metrics
        metrics = {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.get_alpha(),
            'target_q_mean': target_q.mean().item(),
            'current_q1_mean': current_q1.mean().item(),
            'current_q2_mean': current_q2.mean().item(),
        }
        
        if self.tensorboard_log:
            for key, value in metrics.items():
                self.writer.add_scalar(f'training/{key}', value, self.training_step)
        
        return metrics
    
    def soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network."""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'action_bounds': self.action_bounds,
        }
        
        if self.auto_tune_alpha:
            checkpoint['log_alpha'] = self.log_alpha
            checkpoint['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
        
        if self.use_reward_scaling:
            checkpoint['reward_rms_mean'] = self.reward_rms.mean
            checkpoint['reward_rms_var'] = self.reward_rms.var
            checkpoint['reward_rms_count'] = self.reward_rms.count
        
        torch.save(checkpoint, filepath)
        if self.verbose:
            print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        
        self.training_step = checkpoint.get('training_step', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        
        if self.auto_tune_alpha and 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        if self.use_reward_scaling and 'reward_rms_mean' in checkpoint:
            self.reward_rms.mean = checkpoint['reward_rms_mean']
            self.reward_rms.var = checkpoint['reward_rms_var']
            self.reward_rms.count = checkpoint['reward_rms_count']
        
        if self.verbose:
            print(f"Model loaded from {filepath}")
