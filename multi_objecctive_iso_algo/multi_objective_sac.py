import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import copy
import os
from typing import Tuple, Dict, Any, Optional, List
from collections import deque
import random
import gymnasium as gym


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
        
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.BoolTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """Actor network for continuous action spaces."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256],
                 log_std_min: float = -20, log_std_max: float = 2):
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
                 hidden_dims: List[int] = [256, 256]):
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
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        if self.verbose:
            print(f"Using device: {self.device}")
        
        # Multi-objective weights
        if weights is None:
            self.weights = np.ones(reward_dim) / reward_dim  # Equal weights
        else:
            assert len(weights) == reward_dim, f"Weights dimension {len(weights)} != reward_dim {reward_dim}"
            self.weights = np.array(weights)
            self.weights = self.weights / np.sum(self.weights)  # Normalize
        
        # Networks
        self.actor = Actor(state_dim, action_dim, actor_hidden_dims).to(self.device)
        self.critic1 = Critic(state_dim, action_dim, reward_dim, critic_hidden_dims).to(self.device)
        self.critic2 = Critic(state_dim, action_dim, reward_dim, critic_hidden_dims).to(self.device)
        
        # Target networks
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        # Entropy regularization
        self.auto_tune_alpha = auto_tune_alpha
        if auto_tune_alpha:
            if target_entropy is None:
                self.target_entropy = -action_dim
            else:
                self.target_entropy = target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
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
        self.replay_buffer.push(state, action_unscaled, reward, next_state, done)
    
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
        
        self.training_step += 1
        
        # Log to tensorboard
        if self.tensorboard_log and self.training_step % 100 == 0:
            self.writer.add_scalar('Loss/Critic', critic_loss, self.training_step)
            self.writer.add_scalar('Loss/Actor', actor_loss, self.training_step)
            self.writer.add_scalar('Loss/Alpha', alpha_loss, self.training_step)
            self.writer.add_scalar('Alpha', self.get_alpha(), self.training_step)
        
        return {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': self.get_alpha()
        }
    
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
            target_q = rewards + (1 - dones.float().unsqueeze(1)) * self.gamma * target_q
        
        # Current Q-values
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        # Scalarize current Q-values
        current_q1_scalar = self.scalarize_rewards(current_q1)
        current_q2_scalar = self.scalarize_rewards(current_q2)
        
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
        torch.save({
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
            'episode_count': self.episode_count
        }, filepath)
        
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
        
        if self.verbose:
            print(f"Model loaded from {filepath}")


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
            
            if verbose and timesteps % 1000 == 0 and update_info:
                print(f"Step {timesteps}: {update_info}")
        
        # Episode end
        if done:
            episode_rewards.append(episode_reward.copy())
            episode_lengths.append(episode_length)
            
            if verbose:
                print(f"Episode {episode + 1}: Reward {episode_reward}, Length {episode_length}")
            
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
                print(f"Evaluation at step {timesteps}: Mean reward {np.mean(eval_rewards, axis=0)}")
        
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