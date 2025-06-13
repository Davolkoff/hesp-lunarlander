import gymnasium as gym
from network import NeuralNetwork
import numpy as np


def run_with_saved_weights(filename, render=True):
    env = gym.make("LunarLander-v3", render_mode="human" if render else None)
    network = NeuralNetwork(env.observation_space.shape[0], 
                          env.action_space.n, 
                          hidden_units=10)
    
    # Загружаем веса
    network.load_weights(filename)
    
    # Запускаем симуляцию
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = np.argmax(network.predict(state))
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
    
    print(f"Total reward: {total_reward}")
    env.close()