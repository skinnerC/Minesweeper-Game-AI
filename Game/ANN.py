import random as r
import math

class Neuron:
    def __init__(self, value=0.0):
        self.value = value
        
    def useActivationFunct(self, code=0):
        if code == 0:
            self.value = self.sigmoid(self.value)
        else:
            self.value = self.tanh(self.value)
        
    def sigmoid(self, x):
        return 1/(1 + math.exp(-x))

    def tanh(self, x):
        return math.tanh(x)
        
class Synapse:
    def __init__(self, input, output, weight): #input and output are neurons
        self.weight = weight
        self.input = input
        self.output = output
        
    #Set the value of the output neuron to the value of the input neuron*weight
    def setValue(self):
        self.output.value += self.input.value * self.weight
        
class ANN:
    def __init__(self, num_inputs, num_hidden_nodes, num_outputs, weights):
        self.inputLayer = [Neuron() for i in range(num_inputs)] + [Neuron(1.0)]
        self.hiddenLayer = [Neuron() for i in range(num_hidden_nodes)] + [Neuron(1.0)]#
        self.outputLayer = [Neuron() for i in range(num_outputs)]
        self.weights = weights

        i = 0
        # Create input synapses
        self.inputSynapses = []
        for node1 in self.inputLayer:
            for node2 in self.hiddenLayer[:-1]:
                self.inputSynapses.extend([Synapse(node1, node2, weights[i])])
                i += 1

        # Create hidden synapses
        self.hiddenSynapses = []
        for node1 in self.hiddenLayer:
            for node2 in self.outputLayer:
                self.hiddenSynapses.extend([Synapse(node1, node2, weights[i])])
                i += 1
        
        
    def evaluate(self, input):
        self.clearValues()
        # Set the value of the input neuronss
        for i in range(len(self.inputLayer)-1):
            self.inputLayer[i].value = input[i]
            
        # Calculate the values of the neurons in the hidden layer
        for synapse in self.inputSynapses:
            synapse.setValue()

        # This was messed up check its better
        # Uses the activation function on the values of the neurons in the hidden layer
        for node in self.hiddenLayer[:-1]:
            node.useActivationFunct()
            
        # Calculate the values of the neurons in the output layer
        for synapse in self.hiddenSynapses:
            synapse.setValue()
            
        # Uses the activation function on the values of the neurons in the output layer
        for node in self.outputLayer:
            node.useActivationFunct()
        
        return [self.outputLayer[0].value, self.outputLayer[1].value] #[leftTrack, rightTrack]

    def clearValues(self):
        for i in range(len(self.inputLayer)-1):
            self.inputLayer[i].value = 0.0
        for i in range(len(self.hiddenLayer)-1):
            self.hiddenLayer[i].value = 0.0
        for i in range(2):
            self.outputLayer[i].value = 0.0
