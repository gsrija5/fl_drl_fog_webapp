import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import io
import base64

# Deep Q-Network
class DQN(nn.Module):
    def __init__(self, state_dim=5, action_dim=3):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# DRL Agent
class DRLAgent:
    def __init__(self, state_dim=5, action_dim=3, lr=0.001):
        self.model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = []
        self.gamma = 0.95

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, 2)
        with torch.no_grad():
            return torch.argmax(self.model(torch.FloatTensor(state))).item()

    def store(self, experience):
        self.memory.append(experience)
        if len(self.memory) > 1000:
            self.memory.pop(0)

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        q_values = self.model(states).gather(1, actions).squeeze()
        with torch.no_grad():
            max_next_q = self.model(next_states).max(1)[0]
        targets = rewards + self.gamma * max_next_q

        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Federated Averaging for FL
def fed_avg(models):
    global_model = DQN()
    global_dict = global_model.state_dict()

    for key in global_dict.keys():
        global_dict[key] = torch.stack([m.state_dict()[key] for m in models], dim=0).mean(dim=0)

    global_model.load_state_dict(global_dict)
    return global_model

# Unified simulation runner
def run_simulation(agent_type="FL+DRL", rounds=10):
    NUM_NODES = 3
    rewards = []
    latencies = []

    if agent_type == "FL+DRL":
        agents = [DRLAgent() for _ in range(NUM_NODES)]
        for r in range(rounds):
            start = time.time()
            total_reward = 0
            for agent in agents:
                for _ in range(50):
                    state = np.random.rand(5)
                    action = agent.select_action(state)
                    reward = -np.sum(state) * (action + 1)
                    next_state = np.random.rand(5)
                    agent.store((state, action, reward, next_state))
                    agent.train()
                    total_reward += reward

            global_model = fed_avg([a.model for a in agents])
            for agent in agents:
                agent.model.load_state_dict(global_model.state_dict())

            rewards.append(total_reward / (NUM_NODES * 50))
            latencies.append(time.time() - start)

    elif agent_type == "DRL":
        agents = [DRLAgent() for _ in range(NUM_NODES)]
        for r in range(rounds):
            start = time.time()
            total_reward = 0
            for agent in agents:
                for _ in range(50):
                    state = np.random.rand(5)
                    action = agent.select_action(state)
                    reward = -np.sum(state) * (action + 1)
                    next_state = np.random.rand(5)
                    agent.store((state, action, reward, next_state))
                    agent.train()
                    total_reward += reward
            rewards.append(total_reward / (NUM_NODES * 50))
            latencies.append(time.time() - start)

    elif agent_type == "Centralized":
        agent = DRLAgent()
        for r in range(rounds):
            start = time.time()
            total_reward = 0
            for _ in range(NUM_NODES * 50):
                state = np.random.rand(5)
                action = agent.select_action(state)
                reward = -np.sum(state) * (action + 1)
                next_state = np.random.rand(5)
                agent.store((state, action, reward, next_state))
                agent.train()
                total_reward += reward
            rewards.append(total_reward / (NUM_NODES * 50))
            latencies.append(time.time() - start)

    elif agent_type == "Random":
        for r in range(rounds):
            start = time.time()
            total_reward = 0
            for _ in range(NUM_NODES * 50):
                state = np.random.rand(5)
                action = random.randint(0, 2)
                reward = -np.sum(state) * (action + 1)
                total_reward += reward
            rewards.append(total_reward / (NUM_NODES * 50))
            latencies.append(time.time() - start)

    else:
        raise ValueError("Unknown agent_type. Must be FL+DRL, DRL, Centralized, or Random.")

    # Plot the results
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(rewards, marker='o', label=agent_type)
    ax[0].set_title('Efficiency (Avg Reward)')
    ax[0].set_xlabel('Round')
    ax[0].set_ylabel('Average Reward')
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(latencies, marker='x', color='r', label='Latency')
    ax[1].set_title('Latency per Round')
    ax[1].set_xlabel('Round')
    ax[1].set_ylabel('Time (seconds)')
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return image_base64, {
        "reward": round(sum(rewards) / len(rewards), 4),
        "latency": round(sum(latencies) / len(latencies), 4)
    }
