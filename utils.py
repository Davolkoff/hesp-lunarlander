import gymnasium as gym
from network import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt


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

import matplotlib.pyplot as plt
import numpy as np

def visualize_topology(network, gen, filename=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    input_size = len(network.hidden_neurons[0].weights)
    hidden_size = len(network.hidden_neurons)
    output_size = len(network.output_neurons)

    # X-координаты слоёв (вместо Y)
    layer_x = [0, 1, 2]  
    spacing = 1.0  # расстояние между нейронами

    # Y-координаты нейронов
    spacing = 1.0  # расстояние между нейронами

    # Центрируем нейроны по вертикали
    input_y = np.linspace(-spacing*(input_size - 1)/2, spacing*(input_size - 1)/2, input_size)
    hidden_y = np.linspace(-spacing*(hidden_size - 1)/2, spacing*(hidden_size - 1)/2, hidden_size)
    output_y = np.linspace(-spacing*(output_size - 1)/2, spacing*(output_size - 1)/2, output_size)


    # INPUT layer
    for i, y in enumerate(input_y):
        ax.plot(layer_x[0], y, 'o', color='gray', markersize=14, zorder=3)

    # HIDDEN layer + связи от INPUT
    for j, hn in enumerate(network.hidden_neurons):
        y_h = hidden_y[j]
        ax.plot(layer_x[1], y_h, 'o', color='blue', markersize=14, zorder=3)
        for i, w in enumerate(hn.weights):
            y_i = input_y[i]
            linewidth = w
            color = 'green' if w > 0 else 'red'
            ax.plot([layer_x[0], layer_x[1]], [y_i, y_h], linewidth=linewidth, color=color, alpha=0.7, zorder=1)

    # OUTPUT layer + связи от HIDDEN
    for k, on in enumerate(network.output_neurons):
        y_o = output_y[k]
        ax.plot(layer_x[2], y_o, 'o', color='orange', markersize=14, zorder=3)
        for j, w in enumerate(on.weights):
            y_h = hidden_y[j]
            linewidth = max(0.5, min(abs(w)*2, 4))
            color = 'green' if w > 0 else 'red'
            ax.plot([layer_x[1], layer_x[2]], [y_h, y_o], linewidth=linewidth, color=color, alpha=0.7, zorder=1)

    # Добавим легенду
    ax.plot([], [], color='green', label='положительный вес')
    ax.plot([], [], color='red', label='отрицательный вес')
    ax.legend(loc='upper right')

    # Сохраняем
    if filename is None:
        filename = f"topologies/topology_gen_{gen}.png"
    plt.title(f"Топология на поколении {gen}")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_avg_fitness(avg_history):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(avg_history) + 1), avg_history, marker='o', linestyle='-')
    plt.title("Изменение средней приспособленности по поколениям")
    plt.xlabel("Поколение")
    plt.ylabel("Средняя приспособленность (avg)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("avg_fitness_graph.png")
    plt.close()
