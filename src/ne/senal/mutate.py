from random import choice, random, shuffle
from functools import reduce
import  numpy as np
import brain.networks as networks
from brain.elements import Neuron
from brain.elements import Synapse
from brain.runner import NeuralNetworkTask
from utils.properties import Range
import utils.collections as collections


__all__ = ['EhnalMutate']

class EhnalMutate:
    name = 'ehnal_mutate'
    params = {
        'InputNeuronMaxCount' : 20,           # 上游神经元最大个数
        'InputNeuronCountDistribution':[0.0,],
    }
    def __init__(self):
        self.name = EhnalMutate.name