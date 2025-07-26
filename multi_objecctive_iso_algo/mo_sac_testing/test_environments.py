import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from typing import Tuple, Dict, Any
import math


class MultiObjectiveContinuousCartPole(gym.Env):
    """
    Multi-objective version of continuous CartPole environment.
    
    Objectives:
    1. Balance the pole (angle close to 0)
    2. Keep cart position close to center (position close to 0)
    
    State: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    Action: [force] (continuous)
    """
    
    def __init__(self):
        super().__init__()
        
        # Environment parameters
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 30.0
        self.tau = 0.02  # seconds between state updates
        
        # Angle and position thresholds
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        
        # State and action spaces
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max
        ], dtype=np.float32)
        
        self.observation_space = Box(-high, high, dtype=np.float32)
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Multi-objective reward space
        self.reward_space = Box(
            low=np.array([-np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )
        self.reward_dim = 2
        
        self.state = None
        self.steps_beyond_done = None
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict[str, Any]]:
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        
        x, x_dot, theta, theta_dot = self.state
        force = action[0] * self.force_mag
        
        # Physics simulation
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        self.state = (x, x_dot, theta, theta_dot)
        
        # Check termination conditions
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        
        # Multi-objective rewards
        # Objective 1: Keep pole balanced (angle close to 0)
        angle_reward = -abs(theta) * 10
        
        # Objective 2: Keep cart centered (position close to 0)
        position_reward = -abs(x) * 5
        
        # Small survival bonus for both objectives
        if not terminated:
            angle_reward += 1.0
            position_reward += 1.0
        
        reward = np.array([angle_reward, position_reward], dtype=np.float32)
        
        info = {
            'angle': theta,
            'position': x,
            'angle_reward': angle_reward,
            'position_reward': position_reward
        }
        
        return np.array(self.state, dtype=np.float32), reward, terminated, False, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize state with small random values
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        
        return np.array(self.state, dtype=np.float32), {}


class MultiObjectiveMountainCarContinuous(gym.Env):
    """
    Multi-objective version of continuous MountainCar environment.
    
    Objectives:
    1. Reach the goal quickly (minimize time)
    2. Minimize energy consumption (minimize action magnitude)
    
    State: [position, velocity]
    Action: [force] (continuous)
    """
    
    def __init__(self):
        super().__init__()
        
        # Environment parameters
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45
        self.goal_velocity = 0
        self.power = 0.0015
        
        # State and action spaces
        self.low_state = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high_state = np.array([self.max_position, self.max_speed], dtype=np.float32)
        
        self.observation_space = Box(low=self.low_state, high=self.high_state, dtype=np.float32)
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Multi-objective reward space
        self.reward_space = Box(
            low=np.array([-np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )
        self.reward_dim = 2
        
        self.state = None
        self.step_count = 0
        self.max_steps = 999
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict[str, Any]]:
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        
        position, velocity = self.state
        force = action[0]
        
        # Physics simulation
        velocity += force * self.power - 0.0025 * math.cos(3 * position)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        
        # Reset velocity if hit left boundary
        if position == self.min_position and velocity < 0:
            velocity = 0
        
        self.state = np.array([position, velocity], dtype=np.float32)
        self.step_count += 1
        
        # Check goal condition
        goal_reached = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        
        # Check time limit
        time_limit_reached = self.step_count >= self.max_steps
        terminated = goal_reached
        truncated = time_limit_reached
        
        # Multi-objective rewards
        # Objective 1: Reach goal quickly (negative time penalty + goal bonus)
        time_reward = -1.0  # Time penalty
        if goal_reached:
            time_reward += 100.0  # Goal bonus
        
        # Objective 2: Minimize energy consumption
        energy_reward = -abs(force) * 0.1  # Energy penalty
        
        reward = np.array([time_reward, energy_reward], dtype=np.float32)
        
        info = {
            'position': position,
            'velocity': velocity,
            'goal_reached': goal_reached,
            'time_reward': time_reward,
            'energy_reward': energy_reward,
            'step_count': self.step_count
        }
        
        return self.state, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset to random position in valley
        self.state = np.array([
            self.np_random.uniform(low=-0.6, high=-0.4),
            0
        ], dtype=np.float32)
        self.step_count = 0
        
        return self.state, {}


class MultiObjectivePendulum(gym.Env):
    """
    Multi-objective version of Pendulum environment.
    
    Objectives:
    1. Keep pendulum upright (angle close to 0)
    2. Minimize control effort (minimize action magnitude)
    
    State: [cos(theta), sin(theta), angular_velocity]
    Action: [torque] (continuous)
    """
    
    def __init__(self):
        super().__init__()
        
        # Environment parameters
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = 10.0
        self.m = 1.0
        self.l = 1.0
        
        # State and action spaces
        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.observation_space = Box(low=-high, high=high, dtype=np.float32)
        self.action_space = Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        
        # Multi-objective reward space
        self.reward_space = Box(
            low=np.array([-np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )
        self.reward_dim = 2
        
        self.state = None
        self.last_u = None
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict[str, Any]]:
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        
        th, thdot = self.state
        
        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        
        u = action[0]
        u = np.clip(u, -self.max_torque, self.max_torque)
        self.last_u = u
        
        # Physics simulation
        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt
        
        self.state = np.array([newth, newthdot])
        
        # Multi-objective rewards
        # Objective 1: Keep pendulum upright
        # Normalize angle to [-pi, pi]
        angle_normalized = ((newth + np.pi) % (2 * np.pi)) - np.pi
        angle_reward = -(angle_normalized ** 2 + 0.1 * newthdot ** 2)
        
        # Objective 2: Minimize control effort
        control_reward = -(u ** 2) * 0.001
        
        reward = np.array([angle_reward, control_reward], dtype=np.float32)
        
        info = {
            'angle': newth,
            'angular_velocity': newthdot,
            'torque': u,
            'angle_reward': angle_reward,
            'control_reward': control_reward
        }
        
        # Convert state to observation
        obs = np.array([np.cos(newth), np.sin(newth), newthdot], dtype=np.float32)
        
        return obs, reward, False, False, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize with random angle and angular velocity
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        
        # Convert state to observation
        obs = np.array([np.cos(self.state[0]), np.sin(self.state[0]), self.state[1]], dtype=np.float32)
        
        return obs, {}


class MultiObjectiveLunarLander(gym.Env):
    """
    Multi-objective version of LunarLander environment.
    
    Objectives:
    1. Safe landing (minimize distance from landing pad, minimize velocity)
    2. Fuel efficiency (minimize fuel consumption)
    
    This is a simplified 2D version with continuous actions.
    """
    
    def __init__(self):
        super().__init__()
        
        # Environment parameters
        self.world_width = 20.0
        self.world_height = 10.0
        self.landing_pad_x = 0.0
        self.landing_pad_y = 0.0
        self.landing_pad_width = 2.0
        
        # Physics parameters
        self.gravity = -9.8
        self.main_engine_power = 13.0
        self.side_engine_power = 0.6
        self.dt = 0.02
        self.mass = 1.0
        
        # State: [x, y, vx, vy, angle, angular_velocity]
        # Action: [main_engine, left_engine, right_engine] (all continuous [0,1])
        high_obs = np.array([
            self.world_width/2, self.world_height,  # position
            5.0, 5.0,  # velocity
            np.pi, 5.0  # angle, angular velocity
        ], dtype=np.float32)
        
        self.observation_space = Box(-high_obs, high_obs, dtype=np.float32)
        self.action_space = Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Multi-objective reward space
        self.reward_space = Box(
            low=np.array([-np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )
        self.reward_dim = 2
        
        self.state = None
        self.step_count = 0
        self.max_steps = 1000
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict[str, Any]]:
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        
        x, y, vx, vy, angle, angular_velocity = self.state
        
        # Parse actions
        main_engine = action[0]
        left_engine = action[1]
        right_engine = action[2]
        
        # Apply forces
        # Main engine (upward thrust)
        thrust_x = -main_engine * self.main_engine_power * np.sin(angle) / self.mass
        thrust_y = main_engine * self.main_engine_power * np.cos(angle) / self.mass
        
        # Side engines (torque)
        torque = (right_engine - left_engine) * self.side_engine_power
        
        # Update physics
        ax = thrust_x
        ay = thrust_y + self.gravity
        
        vx += ax * self.dt
        vy += ay * self.dt
        x += vx * self.dt
        y += vy * self.dt
        
        angular_velocity += torque * self.dt
        angular_velocity = np.clip(angular_velocity, -5.0, 5.0)
        angle += angular_velocity * self.dt
        
        # Keep angle in reasonable range
        angle = ((angle + np.pi) % (2 * np.pi)) - np.pi
        
        self.state = np.array([x, y, vx, vy, angle, angular_velocity])
        self.step_count += 1
        
        # Check termination conditions
        terminated = False
        landed = False
        crashed = False
        
        if y <= 0:  # Ground contact
            if (abs(x - self.landing_pad_x) <= self.landing_pad_width/2 and
                abs(vx) <= 1.0 and abs(vy) <= 1.0 and abs(angle) <= 0.3):
                landed = True
                terminated = True
            else:
                crashed = True
                terminated = True
        
        # Out of bounds
        if abs(x) > self.world_width/2 or y > self.world_height:
            crashed = True
            terminated = True
        
        # Time limit
        truncated = self.step_count >= self.max_steps
        
        # Multi-objective rewards
        # Objective 1: Safe landing
        landing_reward = 0.0
        if landed:
            landing_reward = 100.0
        elif crashed:
            landing_reward = -100.0
        else:
            # Distance penalty
            distance_to_pad = np.sqrt((x - self.landing_pad_x)**2 + (y - self.landing_pad_y)**2)
            landing_reward = -distance_to_pad * 0.1
            # Velocity penalty
            landing_reward -= (abs(vx) + abs(vy)) * 0.1
            # Angle penalty
            landing_reward -= abs(angle) * 0.1
        
        # Objective 2: Fuel efficiency
        fuel_consumed = main_engine + left_engine + right_engine
        fuel_reward = -fuel_consumed * 0.3
        
        reward = np.array([landing_reward, fuel_reward], dtype=np.float32)
        
        info = {
            'x': x, 'y': y, 'vx': vx, 'vy': vy,
            'angle': angle, 'angular_velocity': angular_velocity,
            'landed': landed, 'crashed': crashed,
            'landing_reward': landing_reward,
            'fuel_reward': fuel_reward,
            'fuel_consumed': fuel_consumed,
            'step_count': self.step_count
        }
        
        return self.state.copy(), reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize at random position high in the sky
        self.state = np.array([
            self.np_random.uniform(-self.world_width/4, self.world_width/4),  # x
            self.np_random.uniform(self.world_height*0.7, self.world_height*0.9),  # y
            self.np_random.uniform(-1.0, 1.0),  # vx
            self.np_random.uniform(-1.0, 1.0),  # vy
            self.np_random.uniform(-0.3, 0.3),  # angle
            0.0  # angular_velocity
        ])
        self.step_count = 0
        
        return self.state.copy(), {}


# Environment registry
MULTI_OBJECTIVE_ENVS = {
    'MultiObjectiveContinuousCartPole-v0': MultiObjectiveContinuousCartPole,
    'MultiObjectiveMountainCarContinuous-v0': MultiObjectiveMountainCarContinuous,
    'MultiObjectivePendulum-v0': MultiObjectivePendulum,
    'MultiObjectiveLunarLander-v0': MultiObjectiveLunarLander
}


def make_env(env_name: str) -> gym.Env:
    """Create a multi-objective environment."""
    if env_name not in MULTI_OBJECTIVE_ENVS:
        raise ValueError(f"Unknown environment: {env_name}. Available: {list(MULTI_OBJECTIVE_ENVS.keys())}")
    
    return MULTI_OBJECTIVE_ENVS[env_name]()


def get_available_envs() -> list:
    """Get list of available multi-objective environments."""
    return list(MULTI_OBJECTIVE_ENVS.keys())
