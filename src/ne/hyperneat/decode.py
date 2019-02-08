#!/usr/bin/python
# -*- coding: UTF-8 -*-

from brain.networks import NeuralNetwork
from utils.properties import Properties

class HyperNEAT:
    def __init__(self,netdef):
        '''
        HyperNEAT
        :param substrate:   list,基座,例如:[(-1, -1), (-1, 0), (-1, 1)],
                                            [(0, -1), (0, 0), (0, 1)],
                                            [(1, 0)]
        '''
        self.netdef = Properties(netdef)
        self.networks = {}

    def createnetwork(self,cppn):
        '''
            :param cppn Individual
            :return: NeuralNetwork
        '''
        net = self.networks.get(cppn.id)
        if net is None:
            net = NeuralNetwork(cppn.id,self.netdef)
            net.createAllNeurons()
            self.networks[cppn.id] = net
        
        return net
    
    def decode(self,ind):
        cppn = ind.genome
        net = self.createnetwork(ind)
        neuronss = net.neurons
        for fromlayer,fromneurons in enumerate(neuronss):            # 每一层
            for fromindex,fromneuron in enumerate(fromneurons):      # 每一层的每一个源神经元
                for tolayer, toneurons in enumerate(neuronss):       # 每一层
                    if fromlayer >= tolayer:continue
                    for toindex,toneuron in enumerate(toneurons):    # 每一层的每一个目标神经元
                        # 用cppn计算权重
                        inputs = [fromneuron.coord.X,fromneuron.coord.Y,toneuron.coord.X,toneuron.coord.Y]
                        output = float(cppn.activate(inputs))
                        # 为网络设置权重
                        synapse = net.getSynapse(fromId=fromneuron.id,toId=toneuron.id)
                        if synapse is None:
                            if output != 0:
                                synapse = net.connect(fromneuron.id,toneuron.id,0,net.definition.models.synapse)
                                synapse['bias'] = output
                        else:
                            if output != 0:
                                synapse['bias'] = output
                            else:
                                net.remove(synapse)
        return net





