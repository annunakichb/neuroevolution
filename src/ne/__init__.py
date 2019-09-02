# -*- coding: UTF-8 -*-

from evolution.agent import IndividualType
import evolution.agent as agent
import  evolution
import brain
from brain.networks import NeuralNetwork

from ne.factory import DefaultNeuralNetworkGenomeFactory

__all__ = ['factory','callbacks','neat','module1']


def neuroevolution_init():
    '''
    神经元进化模块初始化
    :return:
    '''
    # 初始化神经网络模块
    brain.brain_init()
    # 初始化进化模块
    evolution.evolution_init()

    # 注册ne个体类型
    neuralNetworkIndType = IndividualType('network', NeuralNetwork, DefaultNeuralNetworkGenomeFactory(), NeuralNetwork)
    agent.individualTypes.register(neuralNetworkIndType, 'network')


