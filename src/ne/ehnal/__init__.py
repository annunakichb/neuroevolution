import ne
import brain.networks as networks
from brain.networks import NeuralNetwork
from evolution.agent import IndividualType
import evolution.session as session
import evolution.agent as agent

from ne.ehnal import EhnalNeuralNetworkGenomeFactory

import utils.collections as collections
from evolution.idgenerator import NeatIdGenerator
from evolution.selection import NeatSelection
from evolution.mutate import NeatMutate
from evolution.crossmate import NeatCrossmateOperation
from evolution.species import NeatSpeciesMethod

__all__ = ['EhnalNeuralNetworkGenomeFactory','idgenerator','mutate','selection','ehnal_init']


def ehnal_init():
    # 神经元进化模块初始化
    ne.neuroevolution_init()

    # 注册物种划分对象
    agent.speciesType.register(NeatSpeciesMethod(), 'neat_species')
    # 注册选择操作对象
    session.operationRegistry.register(NeatSelection(), 'neat_selection')
    # 注册交叉操作对象
    session.operationRegistry.register(NeatCrossmateOperation(), 'neat_crossmate')
    # 注册变异操作对象
    session.operationRegistry.register(NeatMutate(), 'neat_mutate')

    # 注册操作顺序图
    session.operationGraphis.register('neat_selection,neat_crossmate,neat_mutate', 'neat')
