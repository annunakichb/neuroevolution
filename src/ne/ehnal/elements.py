#!/usr/bin/python3

from brain.elements import NeuralElement
from brain.elements import Neuron

class NeuronBox():
    def __init__(self,id,birth,modelConfiguration,desc='',coord=None):
        self.neurons = []
        self.state['main'] = None
        self.state['follows'] = []
        self.inputs = []
        self.outputs = []
        self.desc = desc

class FeatureNeuron(Neuron):
    def __init__(self,id,layer,birth,modelConfiguration,coord=None):
        Neuron.__init__(self,id,layer,birth,modelConfiguration,coord)
        self.state['activation'] = False
        self.state['liveness'] = 0.
        self.state['features'] = None
        self.state['time'] = 0
        self.box = None


