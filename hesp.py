import pickle
from network import NeuralNetwork, Neuron
import numpy as np


class HESP:
    def __init__(self, env, population_size=50, hidden_units=5, L1_size=10, L2_size=100):
        # Инициализация алгоритма HESP
        self.env = env
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.n
        self.hidden_units = hidden_units
        self.population_size = population_size
        self.L1_size = L1_size
        self.L2_size = L2_size
        
        # Уровень нейронных сетей (L1)
        self.L1 = [NeuralNetwork(self.input_dim, self.output_dim, hidden_units) 
                  for _ in range(L1_size)]
        
        # Уровень нейронов (L2)
        self.L2 = []
        for _ in range(L2_size):
            # Создаем нейроны для скрытого слоя
            hidden_neurons = [Neuron(self.input_dim) for _ in range(hidden_units)]
            # Создаем нейроны для выходного слоя
            output_neurons = [Neuron(hidden_units) for _ in range(self.output_dim)]
            self.L2.extend(hidden_neurons + output_neurons)
        
        self.best_network = None
        self.best_fitness = -np.inf
    
    # 2: Оценивание
    
    def evaluate_network(self, network, episodes=1, render=False):
        # Оценка приспособленности нейронной сети
        total_reward = 0
        
        for _ in range(episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                if render:
                    self.env.render()
                
                action = np.argmax(network.predict(state))
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            total_reward += episode_reward
        
        fitness = total_reward / episodes
        network.fitness = fitness
        
        # Если сеть лучше, чем худшая в L1, обновляем L1
        if len(self.L1) >= self.L1_size and fitness > min(n.fitness for n in self.L1):
            worst_idx = np.argmin([n.fitness for n in self.L1])
            self.L1[worst_idx] = network
        
        # Если сеть лучше лучшей в L1, добавляем ее нейроны в L2
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_network = network
            self.update_L2(network)
            
        return fitness
    
    def update_L2(self, network):
        # Обновление уровня нейронов (L2) на основе лучшей сети
        for neuron in network.hidden_neurons + network.output_neurons:
            if len(self.L2) >= self.L2_size and neuron.fitness > min(n.fitness for n in self.L2):
                worst_idx = np.argmin([n.fitness for n in self.L2])
                self.L2[worst_idx] = neuron
            elif len(self.L2) < self.L2_size:
                self.L2.append(neuron)
    
    def evaluate_population(self, population):
        # Оценка всей популяции
        for network in population:
            if network.fitness == -np.inf:  # Оцениваем только новые сети
                fitness = self.evaluate_network(network)
                # Обновляем лучшую сеть и fitness
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_network = network
    
    # 3: Рекомбинация
    
    def recombine(self):
        # Рекомбинация на уровне нейронных сетей
        new_population = []
        
        # Сортируем сети по приспособленности
        sorted_L1 = sorted(self.L1, key=lambda x: x.fitness, reverse=True)
        
        # Скрещиваем каждую сеть с более приспособленной
        for i in range(len(sorted_L1)):
            parent1 = sorted_L1[i]
            # Выбираем более приспособленного родителя (с меньшим индексом)
            j = max(0, i-1)
            parent2 = sorted_L1[j]
            
            # Создаем потомка с помощью h-точечного кроссинговера
            child = parent1.crossover(parent2)
            child.mutate()
            new_population.append(child)
        
        # Добавляем случайные новые сети
        while len(new_population) < self.population_size:
            new_network = NeuralNetwork(self.input_dim, self.output_dim, self.hidden_units)
            new_population.append(new_network)
        
        return new_population
    
    def evolve(self, generations=100):
        # Основной цикл эволюции
        for gen in range(generations):
            # 2: Оценивание
            self.evaluate_population(self.L1)

            current_best = max(n.fitness for n in self.L1)

            # Отчет о прогрессе
            print(f"Generation {gen+1}, Best Fitness: {current_best}")

            # 3: Рекомбинация
            self.L1 = self.recombine()

            # Ранняя остановка, если достигли хорошего результата
            if current_best >= 200:
                print("Solved!")
                break
            

        
        # Тестирование лучшей сети
        if self.best_network:
            print("Testing best network...")
            self.evaluate_network(self.best_network, episodes=3, render=True)
    
    def save_best_network(self, filename):
        """Сохраняет лучшую сеть в файл"""
        if self.best_network:
            self.best_network.save_weights(filename)
            print(f"Лучшая сеть сохранена в {filename}")
        else:
            print("Нет лучшей сети для сохранения")