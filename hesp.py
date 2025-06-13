import pickle
import numpy as np
import random
from network import NeuralNetwork, Neuron

class HESP:
    def __init__(self, env, population_size=50, hidden_units=5, L1_size=10, L2_size=100):
        self.env = env
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.n
        self.hidden_units = hidden_units
        self.population_size = population_size
        self.L1_size = L1_size
        self.L2_size = L2_size
        self.L2 = []
        self.best_network = None
        self.best_fitness = -np.inf

        # Инициализируем L1: создаем и оцениваем сети
        self.L1 = []
        for _ in range(L1_size):
            net = NeuralNetwork(self.input_dim, self.output_dim, hidden_units)
            self.evaluate_network(net)
            self.L1.append(net)

        

    def evaluate_network(self, network, episodes=5, render=False):
        total_reward = 0
        for _ in range(episodes):
            state, _ = self.env.reset()
            done = False
            while not done:
                if render:
                    self.env.render()
                action = np.argmax(network.predict(state))
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
        fitness = total_reward / episodes
        network.fitness = fitness

        # Обновление L1 и лучшей сети
        if not self.L1:
            self.L1.append(network)
        elif fitness > min(n.fitness for n in self.L1):
            worst = np.argmin([n.fitness for n in self.L1])
            self.L1[worst] = network

        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_network = network
            self.update_L2(network)

        return fitness


    def update_L2(self, network):
        for neuron in network.hidden_neurons + network.output_neurons:
            if len(self.L2) < self.L2_size:
                self.L2.append(neuron)
            elif neuron.fitness > min(n.fitness for n in self.L2):
                worst = np.argmin([n.fitness for n in self.L2])
                self.L2[worst] = neuron

    def evaluate_population(self, population):
        for net in population:
            if net.fitness == -np.inf:
                self.evaluate_network(net)

    def recombine(self):
        new_pop = []
        sorted_L1 = sorted(self.L1, key=lambda x: x.fitness, reverse=True)
        topk = sorted_L1[:max(2, len(sorted_L1)//3)]
        while len(new_pop) < self.population_size:
            p1 = random.choice(topk)
            p2 = random.choice(topk)
            if p2 is p1:
                continue
            child = p1.crossover(p2)
            child.mutate()
            new_pop.append(child)
        return new_pop

    def evolve(self, generations=100):
        for gen in range(generations):
            self.evaluate_population(self.L1)
            
            best = max(n.fitness for n in self.L1)
            avg = np.mean([n.fitness for n in self.L1])
            print(f"Gen {gen+1}: best={best:.2f}, avg={avg:.2f}")

            if avg >= 200:
                print("Solved!")
                break

            self.L1 = self.recombine()
            self.evaluate_population(self.L1)


        if self.best_network:
            print("Testing best...")
            self.evaluate_network(self.best_network, episodes=3, render=True)

    def save_best_network(self, filename):
        if self.best_network:
            self.best_network.save_weights(filename)
            print("Saved to", filename)
        else:
            print("Нет лучшей сети")
