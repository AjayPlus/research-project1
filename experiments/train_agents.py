"""
Script to train and save clean and backdoored agents
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

import torch
from src.environment import EVChargingEnv
from src.agents import DQNAgent, BackdooredDQNAgent


def train_and_save_agent(agent_type='clean', episodes=1000, save_path='checkpoints'):
    """Train and save an agent"""
    os.makedirs(save_path, exist_ok=True)

    # Initialize environment
    env = EVChargingEnv(seed=42)

    # Initialize agent
    if agent_type == 'clean':
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            device='cpu'
        )
        save_file = os.path.join(save_path, 'clean_agent.pt')
    else:
        agent = BackdooredDQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            device='cpu'
        )
        save_file = os.path.join(save_path, 'backdoored_agent.pt')

    print(f"\nTraining {agent_type} agent for {episodes} episodes...")

    rewards = []
    violations = []

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            episode_reward += reward

        rewards.append(episode_reward)
        violations.append(info['violations'])

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            avg_violations = np.mean(violations[-100:])
            print(f"Episode {episode + 1}/{episodes}: "
                  f"Avg Reward={avg_reward:.2f}, Avg Violations={avg_violations:.2f}")

    # Save agent
    agent.save(save_file)
    print(f"\nAgent saved to: {save_file}")

    return agent


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train RL agents')
    parser.add_argument('--agent', type=str, default='clean',
                        choices=['clean', 'backdoored'],
                        help='Agent type to train')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--save_path', type=str, default='checkpoints',
                        help='Directory to save checkpoints')

    args = parser.parse_args()

    train_and_save_agent(args.agent, args.episodes, args.save_path)
