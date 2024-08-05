import gym
import numpy as np
import random

# Initialize the FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=False)

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.1  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate for exploration probability

# Initialize the Q-table
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Training parameters
num_episodes = 1000
max_steps_per_episode = 100

# Q-learning algorithm
for episode in range(num_episodes):
    state, _ = env.reset()  # Use tuple unpacking to get only state
    done = False
    total_reward = 0

    for step in range(max_steps_per_episode):
        # Exploration-exploitation tradeoff
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state, :])  # Exploit

        # Take the action and observe the outcome
        next_state, reward, done, _, _ = env.step(action)

        # Update the Q-table
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state, :])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        state = next_state
        total_reward += reward

        if done:
            break

    # Decay the exploration rate
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Log progress every 100 episodes
    if (episode + 1) % 100 == 0:
        print(f'Episode {episode + 1}/{num_episodes} - Total reward: {total_reward} - Epsilon: {epsilon:.4f}')

    # Print a snapshot of the Q-table
        print(f'Q-table snapshot:\n{q_table}')

# Evaluate the agent
num_eval_episodes = 100
total_rewards = 0

for episode in range(num_eval_episodes):
    state, _ = env.reset()  # Use tuple unpacking to get only state
    done = False
    episode_reward = 0

    for step in range(max_steps_per_episode):
        action = np.argmax(q_table[state, :])  # Always exploit during evaluation
        next_state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        state = next_state

        if done:
            break

    total_rewards += episode_reward

average_reward = total_rewards / num_eval_episodes
print(f'Average reward over {num_eval_episodes} evaluation episodes: {average_reward:.2f}')

env.close()
