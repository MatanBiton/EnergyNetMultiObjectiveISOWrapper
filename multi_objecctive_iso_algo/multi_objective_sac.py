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
    """Experience replay buffer for multi-objective SAC."""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int, reward_dim: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
    
    def push(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, 
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
            torch.from_numpy(rewards_np).float(),
            torch.from_numpy(next_states_np).float(),
            torch.from_numpy(dones_np).bool()
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
    """Critic network for multi-objective rewards."""
    
    def __init__(self, state_dim: int, action_dim: int, reward_dim: int, 
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
        self.output_head = nn.Linear(input_dim, reward_dim)
        
        # Initialize weights
        if use_orthogonal_init:
            self.apply(lambda m: orthogonal_init(m, orthogonal_gain))
            # Use smaller gain for output layer
            orthogonal_init(self.output_head, 0.01)
        else:
            self.apply(xavier_uniform_init)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Q-values for each objective."""
        x = torch.cat([state, action], dim=1)
        x = self.backbone(x)
        return self.output_head(x)


class MultiObjectiveSAC:
    """Multi-Objective Soft Actor-Critic Algorithm."""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 reward_dim: int,
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
                 # Scalarization weights for multi-objective
                 weights: Optional[np.ndarray] = None,
                 # Buffer parameters
                 buffer_capacity: int = 1000000,
                 batch_size: int = 256,
                 # Optimization features (NEW)
                 use_lr_annealing: bool = False,
                 lr_annealing_type: str = 'cosine',  # 'cosine', 'linear', 'exponential'
                 lr_annealing_steps: Optional[int] = None,
                 lr_min_factor: float = 0.1,  # Minimum LR as fraction of initial LR
                 lr_decay_rate: float = 0.95,  # For exponential decay
                 use_reward_scaling: bool = False,
                 reward_scale_epsilon: float = 1e-4,
                 use_orthogonal_init: bool = True,
                 orthogonal_gain: float = 1.0,
                 actor_orthogonal_gain: float = 0.01,
                 critic_orthogonal_gain: float = 1.0,
                 use_value_clipping: bool = False,
                 value_clip_range: float = 200.0,
                 # Logging
                 device: str = 'auto',
                 verbose: bool = False,
                 tensorboard_log: Optional[str] = None):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
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
            if self.device.type == 'cuda':
                print(f"GPU: {torch.cuda.get_device_name(self.device)}")
                print(f"GPU Memory: {torch.cuda.get_device_properties(self.device).total_memory / 1024**3:.1f} GB")
                print(f"CUDA Version: {torch.version.cuda}")
                # Clear GPU cache to start fresh
                torch.cuda.empty_cache()
            else:
                print("WARNING: Using CPU - training will be very slow!")
                print("Make sure CUDA is available and the job requests a GPU.")
            
            # Print optimization features status
            print(f"Optimization features enabled:")
            print(f"  LR Annealing: {self.use_lr_annealing}")
            if self.use_lr_annealing:
                print(f"    Type: {self.lr_annealing_type}")
                print(f"    Steps: {self.lr_annealing_steps}")
                print(f"    Min factor: {self.lr_min_factor}")
            print(f"  Reward Scaling: {self.use_reward_scaling}")
            print(f"  Orthogonal Init: {self.use_orthogonal_init}")
            print(f"  Value Clipping: {self.use_value_clipping}")
            if self.use_value_clipping:
                print(f"    Clip range: {self.value_clip_range}")
        
        # Multi-objective weights
        if weights is None:
            self.weights = np.ones(reward_dim) / reward_dim  # Equal weights
        else:
            assert len(weights) == reward_dim, f"Weights dimension {len(weights)} != reward_dim {reward_dim}"
            self.weights = np.array(weights)
            self.weights = self.weights / np.sum(self.weights)  # Normalize
        
        # Reward scaling statistics
        if self.use_reward_scaling:
            self.reward_rms = RunningMeanStd(shape=(reward_dim,), epsilon=reward_scale_epsilon)
        
        # Networks
        self.actor = Actor(
            state_dim, action_dim, actor_hidden_dims,
            use_orthogonal_init=use_orthogonal_init,
            orthogonal_gain=actor_orthogonal_gain
        ).to(self.device)
        
        self.critic1 = Critic(
            state_dim, action_dim, reward_dim, critic_hidden_dims,
            use_orthogonal_init=use_orthogonal_init,
            orthogonal_gain=critic_orthogonal_gain
        ).to(self.device)
        
        self.critic2 = Critic(
            state_dim, action_dim, reward_dim, critic_hidden_dims,
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
        self.replay_buffer = ReplayBuffer(buffer_capacity, state_dim, action_dim, reward_dim)
        
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
    
    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """Scale action from [-1, 1] to environment bounds."""
        low, high = self.action_bounds
        return low + (action + 1.0) * 0.5 * (high - low)
    
    def unscale_action(self, action: np.ndarray) -> np.ndarray:
        """Unscale action from environment bounds to [-1, 1]."""
        low, high = self.action_bounds
        return 2.0 * (action - low) / (high - low) - 1.0
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray,
                        next_state: np.ndarray, done: bool):
        """Store transition in replay buffer."""
        # Unscale action for storage
        action_unscaled = self.unscale_action(action)
        
        # Apply reward scaling if enabled
        if self.use_reward_scaling:
            # Update reward statistics
            self.reward_rms.update(reward.reshape(1, -1))
            # Store normalized reward
            scaled_reward = self.reward_rms.normalize(reward)
        else:
            scaled_reward = reward
        
        self.replay_buffer.push(state, action_unscaled, scaled_reward, next_state, done)
    
    def scale_rewards(self, rewards: np.ndarray) -> np.ndarray:
        """Scale rewards using running statistics."""
        if self.use_reward_scaling and hasattr(self, 'reward_rms'):
            return self.reward_rms.normalize(rewards)
        return rewards
    
    def unscale_rewards(self, rewards: np.ndarray) -> np.ndarray:
        """Unscale rewards using running statistics."""
        if self.use_reward_scaling and hasattr(self, 'reward_rms'):
            return self.reward_rms.denormalize(rewards)
        return rewards
    
    def scalarize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """Scalarize multi-objective rewards using weights."""
        weights_tensor = torch.FloatTensor(self.weights).to(self.device)
        return torch.sum(rewards * weights_tensor, dim=-1, keepdim=True)
    
    def update(self) -> Dict[str, float]:
        """Update networks."""
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Scalarize rewards for SAC update
        scalar_rewards = self.scalarize_rewards(rewards)
        
        # Update critics
        critic_loss = self._update_critics(states, actions, scalar_rewards, next_states, dones)
        
        # Update actor
        actor_loss = self._update_actor(states)
        
        # Update alpha
        alpha_loss = 0.0
        if self.auto_tune_alpha:
            alpha_loss = self._update_alpha(states)
        
        # Update target networks
        self._update_targets()
        
        # Update learning rate schedulers
        if self.use_lr_annealing:
            self._update_schedulers()
        
        self.training_step += 1
        
        # Log to tensorboard
        if self.tensorboard_log and self.training_step % 100 == 0:
            self.writer.add_scalar('Loss/Critic', critic_loss, self.training_step)
            self.writer.add_scalar('Loss/Actor', actor_loss, self.training_step)
            self.writer.add_scalar('Loss/Alpha', alpha_loss, self.training_step)
            self.writer.add_scalar('Alpha', self.get_alpha(), self.training_step)
            
            # Log learning rates
            if self.use_lr_annealing:
                self.writer.add_scalar('LR/Actor', self.actor_optimizer.param_groups[0]['lr'], self.training_step)
                self.writer.add_scalar('LR/Critic1', self.critic1_optimizer.param_groups[0]['lr'], self.training_step)
                self.writer.add_scalar('LR/Critic2', self.critic2_optimizer.param_groups[0]['lr'], self.training_step)
                if self.auto_tune_alpha:
                    self.writer.add_scalar('LR/Alpha', self.alpha_optimizer.param_groups[0]['lr'], self.training_step)
            
            # Log reward scaling statistics if enabled
            if self.use_reward_scaling and hasattr(self, 'reward_rms'):
                for i in range(self.reward_dim):
                    self.writer.add_scalar(f'RewardScaling/Mean_Obj_{i}', self.reward_rms.mean[i], self.training_step)
                    self.writer.add_scalar(f'RewardScaling/Std_Obj_{i}', np.sqrt(self.reward_rms.var[i]), self.training_step)
        
        return {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': self.get_alpha()
        }
    
    def _update_schedulers(self):
        """Update learning rate schedulers."""
        if self.actor_scheduler is not None:
            self.actor_scheduler.step()
        if self.critic1_scheduler is not None:
            self.critic1_scheduler.step()
        if self.critic2_scheduler is not None:
            self.critic2_scheduler.step()
        if self.alpha_scheduler is not None:
            self.alpha_scheduler.step()
    
    def _update_critics(self, states: torch.Tensor, actions: torch.Tensor, 
                       rewards: torch.Tensor, next_states: torch.Tensor, 
                       dones: torch.Tensor) -> float:
        """Update critic networks."""
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            
            # Use scalarized target Q-values
            target_q1_scalar = self.scalarize_rewards(target_q1)
            target_q2_scalar = self.scalarize_rewards(target_q2)
            target_q = torch.min(target_q1_scalar, target_q2_scalar) - self.get_alpha() * next_log_probs
            
            # Apply value clipping if enabled
            if self.use_value_clipping:
                target_q = torch.clamp(target_q, -self.value_clip_range, self.value_clip_range)
            
            target_q = rewards + (1 - dones.float().unsqueeze(1)) * self.gamma * target_q
        
        # Current Q-values
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        # Scalarize current Q-values
        current_q1_scalar = self.scalarize_rewards(current_q1)
        current_q2_scalar = self.scalarize_rewards(current_q2)
        
        # Apply value clipping to current Q-values if enabled
        if self.use_value_clipping:
            current_q1_scalar = torch.clamp(current_q1_scalar, -self.value_clip_range, self.value_clip_range)
            current_q2_scalar = torch.clamp(current_q2_scalar, -self.value_clip_range, self.value_clip_range)
        
        # Critic losses
        critic1_loss = F.mse_loss(current_q1_scalar, target_q)
        critic2_loss = F.mse_loss(current_q2_scalar, target_q)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        return (critic1_loss + critic2_loss).item() / 2
    
    def _update_actor(self, states: torch.Tensor) -> float:
        """Update actor network."""
        actions, log_probs = self.actor.sample(states)
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        
        # Scalarize Q-values
        q1_scalar = self.scalarize_rewards(q1)
        q2_scalar = self.scalarize_rewards(q2)
        q = torch.min(q1_scalar, q2_scalar)
        
        actor_loss = (self.get_alpha() * log_probs - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def _update_alpha(self, states: torch.Tensor) -> float:
        """Update entropy regularization parameter."""
        with torch.no_grad():
            _, log_probs = self.actor.sample(states)
        
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy)).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        return alpha_loss.item()
    
    def _update_targets(self):
        """Soft update target networks."""
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filepath: str):
        """Save model."""
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_tune_alpha else None,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.auto_tune_alpha else None,
            'weights': self.weights,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            # Optimization parameters
            'use_lr_annealing': self.use_lr_annealing,
            'use_reward_scaling': self.use_reward_scaling,
            'use_orthogonal_init': self.use_orthogonal_init,
            'use_value_clipping': self.use_value_clipping,
            'value_clip_range': self.value_clip_range,
            # Scheduler states
            'actor_scheduler_state_dict': self.actor_scheduler.state_dict() if self.actor_scheduler else None,
            'critic1_scheduler_state_dict': self.critic1_scheduler.state_dict() if self.critic1_scheduler else None,
            'critic2_scheduler_state_dict': self.critic2_scheduler.state_dict() if self.critic2_scheduler else None,
            'alpha_scheduler_state_dict': self.alpha_scheduler.state_dict() if self.alpha_scheduler else None,
            # Reward scaling statistics
            'reward_rms_mean': self.reward_rms.mean if self.use_reward_scaling else None,
            'reward_rms_var': self.reward_rms.var if self.use_reward_scaling else None,
            'reward_rms_count': self.reward_rms.count if self.use_reward_scaling else None,
        }
        
        torch.save(save_dict, filepath)
        
        if self.verbose:
            print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        
        if self.auto_tune_alpha and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        self.weights = checkpoint['weights']
        self.training_step = checkpoint['training_step']
        self.episode_count = checkpoint['episode_count']
        
        # Load scheduler states if they exist
        if self.actor_scheduler and checkpoint.get('actor_scheduler_state_dict'):
            self.actor_scheduler.load_state_dict(checkpoint['actor_scheduler_state_dict'])
        if self.critic1_scheduler and checkpoint.get('critic1_scheduler_state_dict'):
            self.critic1_scheduler.load_state_dict(checkpoint['critic1_scheduler_state_dict'])
        if self.critic2_scheduler and checkpoint.get('critic2_scheduler_state_dict'):
            self.critic2_scheduler.load_state_dict(checkpoint['critic2_scheduler_state_dict'])
        if self.alpha_scheduler and checkpoint.get('alpha_scheduler_state_dict'):
            self.alpha_scheduler.load_state_dict(checkpoint['alpha_scheduler_state_dict'])
        
        # Load reward scaling statistics if they exist
        if self.use_reward_scaling and checkpoint.get('reward_rms_mean') is not None:
            self.reward_rms.mean = checkpoint['reward_rms_mean']
            self.reward_rms.var = checkpoint['reward_rms_var']
            self.reward_rms.count = checkpoint['reward_rms_count']
        
        if self.verbose:
            print(f"Model loaded from {filepath}")
            
            # Print loaded optimization settings
            print("Loaded optimization settings:")
            print(f"  LR Annealing: {checkpoint.get('use_lr_annealing', False)}")
            print(f"  Reward Scaling: {checkpoint.get('use_reward_scaling', False)}")
            print(f"  Orthogonal Init: {checkpoint.get('use_orthogonal_init', True)}")
            print(f"  Value Clipping: {checkpoint.get('use_value_clipping', False)}")
            if checkpoint.get('use_value_clipping', False):
                print(f"    Clip range: {checkpoint.get('value_clip_range', 200.0)}")


def train_mo_sac(env: gym.Env, 
                 agent: MultiObjectiveSAC,
                 total_timesteps: int = 1000000,
                 learning_starts: int = 10000,
                 train_freq: int = 1,
                 eval_freq: int = 10000,
                 eval_episodes: int = 10,
                 save_freq: int = 50000,
                 save_path: str = "mo_sac_model",
                 verbose: bool = False) -> Dict[str, List[float]]:
    """Train Multi-Objective SAC agent."""
    
    # Set annealing steps if not set and annealing is enabled
    if agent.use_lr_annealing and agent.lr_annealing_steps is None:
        agent.lr_annealing_steps = total_timesteps // train_freq
        if agent.verbose:
            print(f"Setting LR annealing steps to {agent.lr_annealing_steps}")
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    timesteps = 0
    episode = 0
    
    state, _ = env.reset()
    episode_reward = np.zeros(agent.reward_dim)
    episode_length = 0
    
    while timesteps < total_timesteps:
        # Select action
        if timesteps < learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)
        
        # Step environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store transition
        agent.store_transition(state, action, reward, next_state, done)
        
        state = next_state
        episode_reward += reward
        episode_length += 1
        timesteps += 1
        
        # Update agent
        if timesteps >= learning_starts and timesteps % train_freq == 0:
            update_info = agent.update()
            
            # GPU memory management
            if agent.device.type == 'cuda' and timesteps % 10000 == 0:
                torch.cuda.empty_cache()  # Clear unused GPU memory periodically
            
            if verbose and timesteps % 1000 == 0 and update_info:
                gpu_info = ""
                if agent.device.type == 'cuda':
                    memory_used = torch.cuda.memory_allocated(agent.device) / 1024**3
                    memory_total = torch.cuda.get_device_properties(agent.device).total_memory / 1024**3
                    gpu_info = f" | GPU: {memory_used:.1f}/{memory_total:.1f} GB"
                
                lr_info = ""
                if agent.use_lr_annealing:
                    actor_lr = agent.actor_optimizer.param_groups[0]['lr']
                    critic_lr = agent.critic1_optimizer.param_groups[0]['lr']
                    lr_info = f" | LR: A={actor_lr:.2e} C={critic_lr:.2e}"
                
                print(f"Step {timesteps}: {update_info}{gpu_info}{lr_info}")
        
        # Episode end
        if done:
            episode_rewards.append(episode_reward.copy())
            episode_lengths.append(episode_length)
            
            if verbose:
                scalarized_reward = np.sum(episode_reward * agent.weights)
                print(f"Episode {episode + 1}: Reward {episode_reward}, Scalarized: {scalarized_reward:.3f}, Length {episode_length}")
            
            # Log to tensorboard
            if agent.tensorboard_log:
                for i, r in enumerate(episode_reward):
                    agent.writer.add_scalar(f'Episode/Reward_Objective_{i}', r, episode)
                agent.writer.add_scalar('Episode/Length', episode_length, episode)
                agent.writer.add_scalar('Episode/Scalarized_Reward', 
                                      np.sum(episode_reward * agent.weights), episode)
            
            state, _ = env.reset()
            episode_reward = np.zeros(agent.reward_dim)
            episode_length = 0
            episode += 1
            agent.episode_count = episode
        
        # Evaluation
        if timesteps % eval_freq == 0 and timesteps >= learning_starts:
            eval_rewards = evaluate_mo_sac(env, agent, eval_episodes, verbose=verbose)
            if verbose:
                mean_rewards = np.mean(eval_rewards, axis=0)
                scalarized_mean = np.mean(np.sum(eval_rewards * agent.weights, axis=1))
                print(f"Evaluation at step {timesteps}: Mean reward {mean_rewards}, Scalarized: {scalarized_mean:.3f}")
        
        # Save model
        if timesteps % save_freq == 0 and timesteps >= learning_starts:
            agent.save(f"{save_path}_{timesteps}.pth")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }


def evaluate_mo_sac(env: gym.Env, 
                    agent: MultiObjectiveSAC, 
                    num_episodes: int = 10,
                    verbose: bool = False) -> np.ndarray:
    """Evaluate Multi-Objective SAC agent."""
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = np.zeros(agent.reward_dim)
        
        while True:
            action = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        
        if verbose:
            print(f"Eval Episode {episode + 1}: Reward {episode_reward}")
    
    return np.array(episode_rewards)