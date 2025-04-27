import pandas as pd
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import joblib

# Battery Environment
class BatteryEnv(gym.Env):
    def __init__(self, data, degradation_model, price_factor=1.0):
        super(BatteryEnv, self).__init__()
        self.data = data.copy()
        self.data['RTM_price'] *= price_factor  # Scale RTM prices for scenarios
        self.degradation_model = degradation_model
        self.current_step = 0
        self.max_steps = len(data) - 1
        self.soc = 50.0
        self.current = 0.0
        self.capacity = 360.0  # Ah
        self.v_nominal = 1344.0  # V
        self.i_max = 74.4  # A
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, -self.i_max, -50.0, 0.0]),
            high=np.array([100.0, self.i_max, 50.0, 1000.0]),
            dtype=np.float32
        )

    def reset(self):
        self.current_step = 0
        self.soc = 50.0
        self.current = 0.0
        return self._get_state()

    def step(self, action):
        temperature = self.data['Cell_Temperature_Average'].iloc[self.current_step]
        rtm_price = self.data['RTM_price'].iloc[self.current_step]

        if action == 0:  # idle
            next_current = 0.0
        elif action == 1:  # charge
            next_current = self.i_max
        else:  # discharge
            next_current = -self.i_max

        delta_charge = next_current * (1 / 60)
        delta_soc = (delta_charge / self.capacity) * 100
        new_soc = max(0.0, min(100.0, self.soc + delta_soc))

        energy = (delta_soc / 100) * (self.capacity * self.v_nominal / 1000) / 1000  # MWh
        profit = 0.0
        if next_current > 0:  # charging
            profit = -energy * rtm_price
        elif next_current < 0:  # discharging
            profit = -energy * rtm_price

        features = np.array([[self.soc, delta_soc, temperature, next_current]])
        degradation = self.degradation_model.predict(features)[0]
        reward = profit - (degradation * 10000.0)

        self.soc = new_soc
        self.current = next_current
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return self._get_state(), reward, done, {}

    def _get_state(self):
        temperature = self.data['Cell_Temperature_Average'].iloc[self.current_step]
        rtm_price = self.data['RTM_price'].iloc[self.current_step]
        return np.array([self.soc, self.current, temperature, rtm_price], dtype=np.float32)

# DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.target_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_states).max(1)[0]
        targets = rewards + self.gamma * next_q_values * (1 - dones)
        loss = nn.MSELoss()(q_values, targets.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

# Standard Strategies
def always_charge_strategy(env):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = 1  # Always charge
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
    return total_reward

def always_discharge_strategy(env):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = 2  # Always discharge
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
    return total_reward

def simple_rtm_vs_dam_strategy(env):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        rtm_price = env.data['RTM_price'].iloc[env.current_step]
        dam_price = env.data['DAM_price'].iloc[env.current_step]
        if rtm_price > dam_price:
            action = 2  # Discharge
        elif rtm_price < dam_price:
            action = 1  # Charge
        else:
            action = 0  # Idle
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
    return total_reward

# Main Function
def main():
    # Load data and RandomForest model
    data = pd.read_csv('synthetic_data.csv')
    degradation_model = joblib.load('degradation_model.pkl')
    print("Loaded synthetic data and RandomForest degradation model.")

    # Compute delta_SOC for model predictions
    data['delta_SOC'] = data['SOC'].diff().fillna(0)

    # Evaluate AI Agent and Strategies Across Scenarios
    scenarios = {'Low Price (0.5x)': 0.5, 'Normal Price (1x)': 1.0, 'High Price (2x)': 2.0}
    results = {scenario: {'AI': [], 'Charge': [], 'Discharge': [], 'Simple': []} for scenario in scenarios}

    for scenario, price_factor in scenarios.items():
        print(scenario,price_factor)
        env = BatteryEnv(data, degradation_model, price_factor=price_factor)
        agent = DQNAgent(state_dim=4, action_dim=3)

        # Train AI agent
        num_episodes = 2
        rewards = []
        for episode in range(num_episodes):
            print(episode)
            state = env.reset()
            total_reward = 0
            done = False
            step = 0
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.store(state, action, reward, next_state, done)
                agent.update()
                state = next_state
                total_reward += reward
                step += 1
                if step % 100 == 0:
                    agent.update_target()
            rewards.append(total_reward)
            print(f"{scenario} - Episode {episode + 1}, Total Reward: {total_reward:.2f}")

        # Evaluate strategies
        ai_reward = rewards[-1]
        charge_reward = always_charge_strategy(env)
        discharge_reward = always_discharge_strategy(env)
        simple_reward = simple_rtm_vs_dam_strategy(env)

        results[scenario]['AI'] = ai_reward
        results[scenario]['Charge'] = charge_reward
        results[scenario]['Discharge'] = discharge_reward
        results[scenario]['Simple'] = simple_reward

    # Save results for plotting
    results_df = pd.DataFrame({
        'Scenario': [s for s in scenarios.keys() for _ in range(4)],
        'Strategy': ['AI', 'Charge', 'Discharge', 'Simple'] * len(scenarios),
        'Reward': [results[s][strat] for s in scenarios for strat in ['AI', 'Charge', 'Discharge', 'Simple']]
    })
    results_df.to_csv('strategy_results.csv', index=False)
    print("Strategy results saved to 'strategy_results.csv'.")

if __name__ == "__main__":
    main()
