import numpy as np
import random


# 1: Инициализация

class Neuron:
    def __init__(self, input_dim):
        # Инициализация весов нейрона случайными значениями
        self.weights = np.random.randn(input_dim) * 0.1
        self.bias = np.random.randn() * 0.1
        self.fitness = -np.inf  # Начальная приспособленность
        
    def activate(self, x):
        # Линейная активация с последующим tanh
        return np.tanh(np.dot(x, self.weights) + self.bias)
    
    def mutate(self, mutation_rate=0.1, mutation_scale=0.2):
        # Мутация весов нейрона
        mask = np.random.rand(*self.weights.shape) < mutation_rate
        self.weights += mask * np.random.randn(*self.weights.shape) * mutation_scale
        if np.random.rand() < mutation_rate:
            self.bias += np.random.randn() * mutation_scale

class NeuralNetwork:
    def __init__(self, input_dim, output_dim, hidden_units):
        # Инициализация нейронной сети с одним скрытым слоем
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.hidden_neurons = [Neuron(input_dim) for _ in range(hidden_units)]
        self.output_neurons = [Neuron(hidden_units) for _ in range(output_dim)]
        self.fitness = -np.inf
        
    def predict(self, x):
        # Прямой проход через сеть
        hidden_output = np.array([neuron.activate(x) for neuron in self.hidden_neurons])
        output = np.array([neuron.activate(hidden_output) for neuron in self.output_neurons])
        return output
    
    def mutate(self):
        # Мутация всех нейронов в сети
        for neuron in self.hidden_neurons + self.output_neurons:
            neuron.mutate()
    
    def crossover(self, other_network):
        # h-точечный кроссинговер ("понейронно")
        child = NeuralNetwork(self.input_dim, self.output_dim, self.hidden_units)
        
        for i in range(len(self.hidden_neurons)):
            if random.random() < 0.5:
                child.hidden_neurons[i] = self.hidden_neurons[i]
            else:
                child.hidden_neurons[i] = other_network.hidden_neurons[i]
                
        for i in range(len(self.output_neurons)):
            if random.random() < 0.5:
                child.output_neurons[i] = self.output_neurons[i]
            else:
                child.output_neurons[i] = other_network.output_neurons[i]
                
        return child

    def save_weights(self, filename):
        # сохранение сети в файл
        weights_dict = {
            'hidden_weights': [n.weights.tolist() for n in self.hidden_neurons],
            'hidden_biases': [n.bias for n in self.hidden_neurons],
            'output_weights': [n.weights.tolist() for n in self.output_neurons],
            'output_biases': [n.bias for n in self.output_neurons]
        }
        import json
        with open(filename, 'w') as f:
            json.dump(weights_dict, f)
    
    def load_weights(self, filename):
        """Загружает веса из файла"""
        import json
        with open(filename, 'r') as f:
            weights_dict = json.load(f)
        
        for i, neuron in enumerate(self.hidden_neurons):
            neuron.weights = np.array(weights_dict['hidden_weights'][i])
            neuron.bias = weights_dict['hidden_biases'][i]
        
        for i, neuron in enumerate(self.output_neurons):
            neuron.weights = np.array(weights_dict['output_weights'][i])
            neuron.bias = weights_dict['output_biases'][i]