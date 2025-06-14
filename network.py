import numpy as np
import random

class Neuron:
    def __init__(self, input_dim):
        self.weights = np.random.uniform(-1.0, 1.0, size=input_dim)
        self.bias = np.random.uniform(-1.0, 1.0)
        self.fitness = -np.inf

    def activate(self, x, use_tanh=True):
        z = np.dot(x, self.weights) + self.bias
        return np.tanh(z) if use_tanh else z

    def mutate(self, mutation_rate=0.1, mutation_scale=0.2):
        mask = np.random.rand(*self.weights.shape) < mutation_rate
        self.weights += mask * np.random.randn(*self.weights.shape) * mutation_scale
        if np.random.rand() < mutation_rate:
            self.bias += np.random.randn() * mutation_scale

    def clone(self):
        clone = Neuron(len(self.weights))
        clone.weights = np.copy(self.weights)
        clone.bias = self.bias
        return clone


class NeuralNetwork:
    def __init__(self, input_dim, output_dim, hidden_units):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.hidden_neurons = [Neuron(input_dim) for _ in range(hidden_units)]
        self.output_neurons = [Neuron(hidden_units) for _ in range(output_dim)]
        self.fitness = -np.inf

    def predict(self, x):
        hidden_output = np.array([n.activate(x) for n in self.hidden_neurons])
        output = np.array([n.activate(hidden_output, use_tanh=False) for n in self.output_neurons])
        return output

    def mutate(self, reuse_prob=0.1, L2_pool=None):
        for i, neuron in enumerate(self.hidden_neurons):
            # С вероятностью reuse_prob заменяем нейрон на копию из L2
            if L2_pool and random.random() < reuse_prob:
                neuron_candidates = L2_pool[i]
                neuron_for_replacement = random.choice(neuron_candidates)

                
                self.hidden_neurons[i] = neuron_for_replacement.clone()
            else:
                neuron.mutate()

        for neuron in self.output_neurons:
            neuron.mutate()


    def crossover(self, other):
        child = NeuralNetwork(self.input_dim, self.output_dim, self.hidden_units)
        # клон нейронов
        for i in range(self.hidden_units):
            parent = self.hidden_neurons[i] if random.random() < 0.5 else other.hidden_neurons[i]
            child.hidden_neurons[i] = parent.clone()
        for i in range(self.output_dim):
            parent = self.output_neurons[i] if random.random() < 0.5 else other.output_neurons[i]
            child.output_neurons[i] = parent.clone()
        return child

    def save_weights(self, filename):
        weights = {
            'hidden_weights': [n.weights.tolist() for n in self.hidden_neurons],
            'hidden_biases': [n.bias for n in self.hidden_neurons],
            'output_weights': [n.weights.tolist() for n in self.output_neurons],
            'output_biases': [n.bias for n in self.output_neurons]
        }
        import json
        with open(filename, 'w') as f:
            json.dump(weights, f)

    def load_weights(self, filename):
        import json
        with open(filename, 'r') as f:
            w = json.load(f)
        for i, n in enumerate(self.hidden_neurons):
            n.weights = np.array(w['hidden_weights'][i])
            n.bias = w['hidden_biases'][i]
        for i, n in enumerate(self.output_neurons):
            n.weights = np.array(w['output_weights'][i])
            n.bias = w['output_biases'][i]
