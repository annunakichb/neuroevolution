

__all__ = ['crossmate','idgenerator','mutate','selection','neat_init', 'config']

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
from .species import NeatSpeciesMethod

from ne.factory import DefaultNeuralNetworkGenomeFactory
import ne

def neat_init():
    ne.neuroevolution_init()

    # 注册id生成器对象
    networks.idGenerators.register(NeatIdGenerator(),'neat')
    # 注册物种划分对象
    agent.speciesType.register(NeatSpeciesMethod(),'neat_species')
    # 注册选择操作对象
    session.operationRegistry.register(NeatSelection(),'neat_selection')
    # 注册交叉操作对象
    session.operationRegistry.register(NeatCrossmateOperation(),'neat_crossmate')
    # 注册变异操作对象
    session.operationRegistry.register(NeatMutate(),'neat_mutate')

    # 注册操作顺序图
    session.operationGraphis.register('neat_selection,neat_crossmate,neat_mutate','neat')


