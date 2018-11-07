

__all__ = ['crossmate','idgenerator','mutate','selection','neat_init']

import brain.networks as networks
from brain.networks import NeuralNetwork
from evolution.agent import IndividualType
import evolution.session as session
import evolution.agent as agent

import utils.collections as collections
from .idgenerator import NeatIdGenerator
from .selection import NeatSelection
from .mutate import NeatMutate
from .crossmate import NeatCrossmateOperation

from ne.factory import DefaultNeuralNetworkGenomeFactory

def neat_init():

    # 注册ne个体类型
    neuralNetworkIndType = IndividualType('network', NeuralNetwork, DefaultNeuralNetworkGenomeFactory(), NeuralNetwork)
    agent.individualTypes.register(neuralNetworkIndType, 'network')

    # 注册id生成器对象
    networks.idGenerators.register(NeatIdGenerator(),'neat')
    # 注册选择操作对象
    session.operationRegistry.register(NeatSelection(),'neat_selection')
    # 注册交叉操作对象
    session.operationRegistry.register(NeatCrossmateOperation(),'neat_crossmate')
    # 注册变异操作对象
    session.operationRegistry.register(NeatMutate(),'neat_mutate')

    # 注册操作顺序图
    session.operationGraphis.register('neat_selection,neat_crossmate,neat_mutate','neat')


