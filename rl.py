import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import cv2

# Hyperparameters
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY_STEPS = 2_000_000
BATCH_SIZE = 32
MEMORY_SIZE = 100_000
LEARNING_RATE = 0.00025
TARGET_UPDATE = 10_000
TRAINING_STEPS = 50_000_000
FRAME_STACK = 4
R1, R2 = 4, 20
HIDDEN_LAYER_SIZE = 1024
TEST_EPOCH_STEPS = 125_000

def preprocess_frame(frame):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    elif len(frame.shape) != 2:
        raise ValueError(f"Unexpected frame shape: {frame.shape}")
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    frame = frame / 255.0
    return frame

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, HIDDEN_LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_SIZE, num_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).reshape(x.size(0), -1)
        return self.fc(conv_out)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)

class DFDQNAgent:
    def __init__(self, env, device):
        self.env = env
        self.device = device
        self.original_actions = env.action_space.n
        self.num_actions = self.original_actions * 2
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.q_network = DQN((FRAME_STACK, 84, 84), self.num_actions).to(device)
        self.target_network = DQN((FRAME_STACK, 84, 84), self.num_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        self.epsilon = EPSILON_START
        self.step_count = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        state = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()

    def get_action_repetition(self, action):
        return R1 if action < self.original_actions else R2

    def get_base_action(self, action):
        return action % self.original_actions

    def step(self, state):
        action = self.select_action(state)
        repetitions = self.get_action_repetition(action)
        base_action = self.get_base_action(action)
        total_reward = 0
        next_state = state.copy()
        done = False

        for _ in range(repetitions):
            next_frame, reward, done, _, _ = self.env.step(base_action)
            total_reward += reward
            next_state[:, :, :-1] = next_state[:, :, 1:]
            next_state[:, :, -1] = preprocess_frame(next_frame)
            if done:
                break

        self.memory.push(state, action, total_reward, next_state, done)
        self.step_count += 1
        self.epsilon = max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * self.step_count / EPSILON_DECAY_STEPS)
        return next_state, total_reward, done, action

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        state, action, reward, next_state, done = self.memory.sample(BATCH_SIZE)
        state = torch.FloatTensor(state).permute(0, 3, 1, 2).to(self.device)
        next_state = torch.FloatTensor(next_state).permute(0, 3, 1, 2).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        q_values = self.q_network(state).gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(next_state).max(1)[0]
            target = reward + (1 - done) * GAMMA * next_q_values

        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.step_count % TARGET_UPDATE == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

def main():
    env = gym.make('Seaquest-v4')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DFDQNAgent(env, device)

    state = np.zeros((84, 84, FRAME_STACK))
    observation = env.reset()[0]
    for i in range(FRAME_STACK):
        state[:, :, i] = preprocess_frame(observation)
    episode_reward = 0
    episode_count = 0
    episode_rewards = []

    for step in range(TRAINING_STEPS):
        next_state, reward, done, action = agent.step(state)
        episode_reward += reward
        state = next_state

        agent.train()

        if done:
            observation = env.reset()[0]
            state = np.zeros((84, 84, FRAME_STACK))
            for i in range(FRAME_STACK):
                state[:, :, i] = preprocess_frame(observation)
            episode_rewards.append(episode_reward)
            print(f"Episode {episode_count + 1}, Reward: {episode_reward}")
            episode_reward = 0
            episode_count += 1

        if (step + 1) % TEST_EPOCH_STEPS == 0:
            test_rewards = []
            test_state = np.zeros((84, 84, FRAME_STACK))
            test_observation = env.reset()[0]
            for i in range(FRAME_STACK):
                test_state[:, :, i] = preprocess_frame(test_observation)
            test_episode_reward = 0
            test_steps = 0

            while test_steps < TEST_EPOCH_STEPS:
                action = agent.select_action(test_state)
                repetitions = agent.get_action_repetition(action)
                base_action = agent.get_base_action(action)
                for _ in range(repetitions):
                    next_frame, reward, done, _, _ = env.step(base_action)
                    test_episode_reward += reward
                    test_state[:, :, :-1] = test_state[:, :, 1:]
                    test_state[:, :, -1] = preprocess_frame(next_frame)
                    test_steps += 1
                    if done or test_steps >= TEST_EPOCH_STEPS:
                        break
                if done:
                    test_rewards.append(test_episode_reward)
                    test_observation = env.reset()[0]
                    test_state = np.zeros((84, 84, FRAME_STACK))
                    for i in range(FRAME_STACK):
                        test_state[:, :, i] = preprocess_frame(test_observation)
                    test_episode_reward = 0
                    if test_steps >= TEST_EPOCH_STEPS:
                        break
            avg_test_reward = np.mean(test_rewards) if test_rewards else 0
            print(f"Test Epoch at Step {step + 1}, Average Reward: {avg_test_reward}")

    env.close()

if __name__ == "__main__":
    main()
